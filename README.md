# Beyond FA

<img src="https://github.com/MASILab/beyondFA_mlp/blob/main/4.png" alt="Challenge Logo" width="300">

## Building the Docker

To build this Docker container, clone the repository and run the following command in the root directory:

```bash
DOCKER_BUILDKIT=1 sudo docker build -t beyond_fa_infonce:v1.0.0 .
```

The Docker runs the code from `scripts/entrypoint.sh`.

## Running the Docker

Your Docker container should be able to read input data from `/input` and write output data to `/output`. Intermediate data should be written to `/tmp`. The input data will be a `.mha` file containing the diffusion MRI data with gradient table information contained in a `.json` file. The input file will be in `/input/images/dwi-4d-brain-mri/`, with gradient table information at `/input/dwi-4d-acquisition-metadata.json`. Your Docker should write a JSON list to the output directory with the name `/output/features-128.json`. **Your JSON list must contain 128 values. You may zero-pad the list if you wish to provide fewer than 128 values.**

To run this Docker:

```bash
input_dir="/home-local/lij112/codes/beyond_fa_challenge/results/Inputs"
output_dir="/home-local/lij112/codes/beyond_fa_challenge/results/Outputs-infonce"

mkdir -p $output_dir
sudo chmod 777 $output_dir

DOCKER_NOOP_VOLUME="beyond_fa_pca-volume"
sudo docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
sudo docker run \
    -it \
    --platform linux/amd64 \
    --network none \
    --gpus all \
    --rm \
    --volume $input_dir:/input:ro \
    --volume $output_dir:/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    beyond_fa_infonce:v1.0.0
sudo docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
```
