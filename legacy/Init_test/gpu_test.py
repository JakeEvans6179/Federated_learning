import os
import ray
import tensorflow as tf

print("Driver TF GPUs:", tf.config.list_physical_devices("GPU"))

ray.shutdown()
ray.init(ignore_reinit_error=True, num_cpus=16, num_gpus=1)

print("Ray available resources:", ray.available_resources())

@ray.remote(num_gpus=1)
def gpu_probe():
    import os
    import ray
    import tensorflow as tf
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ray_gpu_ids": ray.get_gpu_ids(),
        "tf_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }

print(ray.get(gpu_probe.remote()))
ray.shutdown()