from fastapi import FastAPI, UploadFile, File
import json
from PIL import Image
from io import BytesIO
import sys

import pymc3 as pm
from pymc3 import *
import theano
import theano.tensor as tt

import numpy as np
from scipy import stats
from matplotlib import pylab as plt
import seaborn as sns
sns.set_style("whitegrid")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from starlette.responses import StreamingResponse



app = FastAPI()



@app.post("/generate_image")
async def generate_image(file: UploadFile = File(...)):
    contents = await file.read()
    data = json.loads(contents)

    price_demand = data['price_demand']

    cost = data['cost']

    #price_demand = {15:20,14:18,13:35,12:50,11:65}
    p0,d0 = list(price_demand.keys()), list(price_demand.values())
    p0 = [int(i) for i in p0]
    
    # Process the JSON data and generate the image

    with pm.Model() as m:

        log_b = pm.Normal('log_b', sd = 5)                # priors
        c = pm.HalfNormal('c', sd = 5)                    # assume the elasticty to be non-negative

        log_d = log_b - c * np.log(p0)                    # demand model
        pm.Poisson('d0', np.exp(log_d), observed = d0)    # likelihood

        s = pm.sample(1000,tune=500, cores=1)     

        p = np.linspace(10, 16)   # price range
        print(p.shape)
        print(p.reshape(-1,1).shape)
        #d_means = np.exp(s.log_b - s.c * np.log(p).reshape(-1, 1))[:, :500]
        d_means = np.exp(s.log_b - s.c * np.log(p).reshape(-1, 1))[:, :500]
        print(d_means.shape)




        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(p, d_means, c = 'k', alpha = 0.01)
        plt.plot(p0, d0, 'ko', markeredgewidth=1.5, markerfacecolor='w', markersize=10)
        plt.xlabel('Price ($)')
        plt.ylabel('Demand (Units)')

        profit = (p - cost).reshape(-1,1) * d_means
        pmax = p[np.argmax(np.mean(profit,1))]
        print(pmax)

        # Create an in-memory stream
        image_stream = BytesIO()
        plt.savefig(image_stream, format="png")
        plt.close(fig)
        
        # Set the stream position to the beginning
        image_stream.seek(0)

        #image_bytes = BytesIO()
        #plt.savefig(image_bytes)
        #plt.show()
        



        #sys.exit()
        #image_bytes.seek(0)
        #img = Image.open(image_bytes)
    
    #return {"image":  img }
    return StreamingResponse(image_stream, media_type="image/png")

    
    #image = Image.new('RGB', (width, height), color)
    
    # Convert the generated image to bytes


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
