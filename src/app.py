import logging
import os
import uvicorn

from fastapi import FastAPI

from real_estate_app import mount_realestate_app

# Log a message using the custom logger
logging.info("Logging initialized.")

# Set the environment variables for the environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "Local")

# Start the FastAPI app
app = FastAPI()

#  Mounts sub-applications for modular routing.
app = mount_realestate_app(app)

# Log a message using the custom logger
logging.info("FastAPI app started.")
# Log the root path and routes of the FastAPI app
logging.info(app.root_path)
logging.info(app.routes)

if __name__ == "__main__":

    logging.info("Starting Real Estate app with Uvicorn...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
        proxy_headers=True,
    )
