# Cost analysis of Air Pressure System failure.

### Problem Statement

The system in focus is the Air Pressure system (APS) which generates pressurised air that are utilized to force a piston to provide pressure to the brake pads, slowing the vehicle down and also in gear changes. The benefits of using an APS instead of a hydraulic system are the easy availability and long-term sustainability of natural air.

The datasets' affirmative class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures 
for components not related to the APS. This is a Binary Classification problem.

### Solution Proposed 
-- Challenge metric  

     Cost-metric of miss-classification:

     Predicted class |      True class       |
                     |    pos    |    neg    |
     -----------------------------------------
      pos            |     -     |  Cost_1   |
     -----------------------------------------
      neg            |  Cost_2   |     -     |
     -----------------------------------------
     Cost_1 = 10 and cost_2 = 500

The total cost of a prediction model the sum of "Cost_1" multiplied by the number of Instances with type 1 failure and "Cost_2" with the number of instances with type 2 failure, resulting in a "Total_cost".

In this case Cost_1 refers to the cost that an unnessecary check needs to be done by an mechanic at an workshop, while Cost_2 refer to the cost of missing a faulty truck, which may cause a breakdown.

     Total_cost = Cost_1*No_Instances + Cost_2*No_Instances.
The problem is to reduce the cost due to unnecessary repairs. So it is required to minimize the false predictions.

### Dataset information
The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class and number of Attributes is 171.

## Tech Stack Used

1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions

## How to run?

Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Data Collections

![image](https://user-images.githubusercontent.com/57321948/193536736-5ccff349-d1fb-486e-b920-02ad7974d089.png)

## Project Archietecture

![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)

## Deployment Archietecture

![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)

### Step 1: Clone the repository

```bash
git clone https://github.com/MaheshKumarMK/Cost-Analysis-of-Air-pressure-system.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.7.6 -y
```

```bash
conda activate venv
```

### Step 3 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 4 - Export the environment variable

```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL>

```

### Step 5 - Run the application server

```bash
python main.py
```

### Step 6. Train application

```bash
http://localhost:8080/train

```

### Step 7. Prediction application

```bash
http://localhost:8080/predict

```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build -t sensor . 

```

3. Run the Docker image

```
docker run -d -e AWS_ACCESS_KEY_ID="${{ secrets.AWS_ACCESS_KEY_ID }}" -e AWS_SECRET_ACCESS_KEY="${{ secrets.AWS_SECRET_ACCESS_KEY }}" -e AWS_DEFAULT_REGION="${{ secrets.AWS_DEFAULT_REGION }}" -e MONGODB_URL="${{ secrets.MONGODB_URL }}" -p 8080:8080 sensor
```
