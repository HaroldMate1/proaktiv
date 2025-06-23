# Use a Miniconda3 base image as a starting point
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file to the container
COPY environment.yml .

# Create the Conda environment from the .yml file
# This installs all necessary packages, including Python and CUDA toolkit
RUN conda env create -f environment.yml

# Make Conda's `activate` command available to the shell
SHELL ["conda", "run", "-n", "proaktiv", "/bin/bash", "-c"]

# Display Conda environment details for verification
RUN echo "Conda environment 'proaktiv' created." && \
    conda activate proaktiv && \
    conda list

# Copy the rest of the project source code into the container
COPY . .

# Activate the Conda environment for all subsequent commands
# This ensures that our scripts run with the correct Python interpreter and packages
ENV PATH /opt/conda/envs/proaktiv/bin:$PATH

# Default command to run when the container starts
# For example, to run the inference script for the mutation test
# This can be overridden from the docker run command line
CMD ["python", "src/inference/predict.py", "--help"]

