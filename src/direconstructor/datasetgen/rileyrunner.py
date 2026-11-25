import os

#Prevent numerical libraries reserving large thread pools/address ranges.
#Must run before numpy/riley are imported, or it has no effect in this process.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import gc
import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import riley  # type: ignore[reportMissingImports]


class RileyRunner:
    """
    Renders a single speckle image with riley.

    args must provide (attribute access, e.g. argparse.Namespace / SimpleNamespace):
        pattern, realization, coverage, scale, side,
        width, height, focal_length, fov_scale, world_rot, roi_cent_world, sub_sample,
        effects_seed, k1, k2, k3, p1, p2, blur_sigma  (each a (MIN, MAX) pair)

    Usage (one runner per pattern; mesh, base camera and rasterizer are reused):
        runner = RileyRunner(args, directory, "clean")
        runner.set_rasterizer()
        runner.generate_pattern()
        runner.set_camera(distortion=False)
        runner.run()
        for variant in range(n):
            runner.name = f"observed_{variant:03d}"   #Set name BEFORE set_camera (it seeds the RNG).
            runner.set_camera(distortion=True)
            runner.run()
        runner.release()
    """

    def __init__(self, args, directory, name):
        self.args = args
        self.directory = Path(directory)
        self.name = name

        self.base_camera = None
        self.camera = None
        self.config = None
        self.mesh = None
        self.distortion_parameters = None

    def write_settings(self):
        """
        Writes riley settings into a .json file.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        settings = vars(self.args).copy()
        settings.update(
            render_threads = 1,
            riley_module = str(riley.__file__),
            numpy_version = np.__version__,
        )

        #Write into json.
        (self.directory / "settings.json").write_text(
            json.dumps(settings, indent=4, default=str) + "\n", encoding="utf-8"
        )

    def set_camera(self, distortion = False):
        #Set the perfect camera once; distorted cameras are copies of it.
        if self.base_camera is None:
            focal_length = self.args.focal_length
            self.base_camera = riley.Camera(
                    pixels_num=(self.args.width, self.args.height),
                    pixels_size=(focal_length / self.args.width, focal_length / self.args.height),
                    pos_world=(0.0, 0.0, self.args.side * self.args.fov_scale),
                    rot_world=self.args.world_rot,
                    roi_cent_world=self.args.roi_cent_world,
                    focal_length=focal_length,
                    sub_sample=self.args.sub_sample,
                )
        camera = self.base_camera

        if(distortion):
            #Seed RNG for distortion. self.name is included so each variant of the
            #same pattern gets different effects, independent of render order.
            rng = random.Random(
                        f"{self.args.effects_seed}:{self.args.pattern}:{self.args.realization}:{self.args.coverage}:{self.args.scale}:{self.name}"
                    )

            #Distortion model coefficients.
            coefficients = {
                f"distort_{coefficient}": rng.uniform(*getattr(self.args, coefficient))
                for coefficient in ["k1", "k2", "k3", "p1", "p2"]
            }

            sigma = rng.uniform(*self.args.blur_sigma)

            #Collect all distortion parameters. Must be flat: every key is a riley.Camera field.
            self.distortion_parameters = dict(
                coefficients,
                distort_model = int(any(coefficients.values())),
                psf_type = int(sigma > 0),
                psf_sigma_x = sigma,
                psf_support_rad = 3 * sigma,
                psf_separable = 1,
            )

            self.camera = replace(camera, **self.distortion_parameters)
        else:
            self.distortion_parameters = None
            self.camera = camera

    def set_rasterizer(self):
        #Set rasterizer.
        self.config = riley.RasterConfig(
            parallel=1,
            save_strategy=riley.SaveStrategy.disk,
            image_save_mode=riley.ImageSaveMode.multifield,
            save_format=riley.ImageFormat.bmp,
            save_bits=8,
            save_scaling=riley.ScaleStrategy.fixed,
            save_scaling_min=0.0,
            save_scaling_max=1.0,
            report=riley.ReportMode.off,
            output_name_format=self.name, #Overridden per image in run().
            background_value=1.0,
        )

    def generate_pattern(self):
        #Setup a big fat square of a mesh.
        h = self.args.side / 2
        coords = np.array(((-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)))
        connect = np.array(((0, 1, 2, 3),), dtype=np.int64) #Connectivity map for the mesh.
        uvs = np.array(((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
        convention = riley.ConnectConvention(
            riley.EElemType.QUAD4,
            riley.EConnectAxis.ROW,
            0,
            riley.ENodeOrder.RILEY,
        )

        #Speckle parameters. Perlin uses scale as cell size, dots use it as radius.
        if self.args.pattern == "perlin":
            params = riley.Perlin2DParams(seed=self.args.realization, cell_size=self.args.scale)
        else:
            params = riley.Dot2DParams(
                seed=self.args.realization,
                coverage=self.args.coverage,
                radius=self.args.scale,
                pattern=riley.SpecklePattern[self.args.pattern],
            )

        shader = riley.SpeckleShader(
            uvs=uvs,
            params=params,
            bits=-1,
            scaling_type=riley.ScaleStrategy.none,
        )
        self.mesh = riley.create_mesh(
            convention, riley.MeshType.quad4, coords, connect, shader
        )

    def write_metadata(self, image):
        """
        Writes <name>.json next to the rendered image and returns the record.
        """
        is_perlin = self.args.pattern == "perlin"
        record = {
            "pattern": self.args.pattern,
            "seed": self.args.realization,
            "coverage": None if is_perlin else self.args.coverage,
            "radius": None if is_perlin else self.args.scale,
            "cell_size": self.args.scale if is_perlin else None,
            "image": image.name,
            "distorted": self.distortion_parameters is not None,
            "camera": asdict(self.camera),
            "width": self.args.width,
            "height": self.args.height,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
        (self.directory / f"{self.name}.json").write_text(
            json.dumps(record, indent=4, default=str) + "\n", encoding="utf-8"
        )
        return record

    def run(self):
        """
        Renders the current camera's image as <name>.bmp, writes its metadata, and
        returns the metadata record (e.g. for the generator to append to a manifest).
        """
        if self.camera is None or self.config is None or self.mesh is None:
            raise RuntimeError("call set_camera, set_rasterizer and generate_pattern before run")

        self.directory.mkdir(parents=True, exist_ok=True)

        #Project the speckle pattern onto the mesh through the camera.
        config = replace(self.config, output_name_format=self.name)
        riley.raster(self.mesh, self.camera, config, out_dir=str(self.directory))
        image = self.directory / f"{self.name}.bmp"

        return self.write_metadata(image)

    def release(self):
        #Free the native mesh; massive runs otherwise accumulate memory.
        self.mesh = None
        gc.collect()


if __name__ == "__main__":
    from types import SimpleNamespace

    #Example of what the run plan generator loop looks like.
    args = SimpleNamespace(
        pattern="disk",
        realization=1,
        coverage=0.5,
        scale=0.25,
        side=40.0,
        width=512,
        height=512,
        focal_length=50e-3,
        fov_scale=1.2,
        world_rot=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        sub_sample=4,
        effects_seed=42,
        k1=(-0.2, 0.2),
        k2=(-0.03, 0.03),
        k3=(0.0, 0.0),
        p1=(-0.001, 0.001),
        p2=(-0.001, 0.001),
        blur_sigma=(0.4, 1.2),
    )
    directory = Path("out/test")

    runner = RileyRunner(args, directory, "clean")
    runner.write_settings()
    runner.set_rasterizer()
    runner.generate_pattern()

    runner.set_camera(distortion=False)
    print(runner.run()["image"])

    for variant in range(2):
        runner.name = f"modified_{variant:03d}"
        runner.set_camera(distortion=True)
        print(runner.run()["image"])

    runner.release()
