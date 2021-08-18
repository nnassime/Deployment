#deep learning libraries
from fastai.vision import *
import torch
defaults.device = torch.device('cpu')

#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio

import os
import sys
import base64 
from PIL import Image

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()
#path = Path('')
#learner = load_learner(path)

@app.route("/upload", methods = ["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods = ["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):

    # Added ******************

    img = io.BytesIO(bytes)
    img.seek(0)
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    print('imag from cv2e read')
    model="yolov3-tiny"
    confidence=0.45

    bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model=model)



    output_image = draw_bbox(frame, bbox, label, conf, write_conf=True)
    print('prediction object detection  donne')
    print(label, conf)

    pred_class = label

    cv2.imwrite('img_output.jpg', output_image) 
    img_uri = base64.b64encode(open("img_output.jpg", 'rb').read()).decode('utf-8')

    # Added ******************



    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in conf]]
    pred_probs = sorted(zip(label, map(str, formatted_outputs)),
                        key = lambda p: p[1],
                        reverse = True
                       )
    return HTMLResponse(
        """
        <html>
            <body>
                <p> Prediction: <b> %s </b> </p>
                <p> Confidence: <b> %s </b> </p>
            </body>
        <figure class = "figure">
            <img src="data:image/png;base64, %s" class = "figure-img">
        </figure>
        </html>
        """ %(pred_class, pred_probs, img_uri))
        
@app.route("/")
def form(request):
        return HTMLResponse(
            """
            <h1> Greenr </h1>
            <p> Deployment of Yolov3.tiny by Data Science and Applications Research Unit </p>
            <form action="/upload" method = "post" enctype = "multipart/form-data">
                <u> Select picture to upload: </u> <br> <p>
                1. <input type="file" name="file"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            <br>
            <br>
            <u> Submit picture URL </u>
            <form action = "/classify-url" method="get">
                1. <input type="url" name="url" size="60"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            """)
        
@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")
        
if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)
