1. Detailed Component Descriptions

A. API Gateway & Upload Service: 
As the only entry point of the system, it handles authentication (API Key), Rate Limiting, and request routing.

B. Dataset processor
Verification: Check for image corruption and minimum resolution requirements 
Captioning: Run the BLIP or LLaVA model to generate descriptive text.
Cleanup: Automatically remove EXIF information and convert to .jpg or .png format.
Enhancement: Perform random cropping or proportional alignment based on configuration.

C.GPU Worker Pool
Resource locking: Each Worker process is bound with a CUDA_VISIBLEDEVICES number at startup.
Dynamic scaling preparation: Although there are currently only 2-4 blocks, through containerization deployment (K8s Device Plugin), adding nodes in the future only requires modifying the number of replicas.

D. Model Evaluation Service & Quality Gate
Duties: Perform 'automated acceptance' before deployment.
CLIP Score: Evaluate the semantic consistency between the generated image and the Prompt.
Aesthetic Score: Evaluating the aesthetic score of an image using a pre trained classifier

2. API specification

Endpoint: POST /v1/datasets：Initialize dataset upload.
Endpoint: POST /v1/train：Start the training assignment.
Endpoint: GET /v1/jobs/{job_id}：Check the progress and logs of homework.

3. Data Models & Schemas
Model metadata can be stored in Model Registry.

4. Error Handling Strategies
To maximize 2–4 GPU utilization, the system implements aggressive error detection and automated recovery.
#. Compute Errors (Worker Level)
##. Quality Gating (Deployment Level)

