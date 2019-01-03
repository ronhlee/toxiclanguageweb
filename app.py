from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import aiohttp, asyncio

from fastai import *
from fastai.text import *

model_file_name = 'model'
model_file_url = 'https://dl.dropboxusercontent.com/s/b6uwmglcga1uuox/first.pth?dl=0'
train_ids_url = 'https://dl.dropboxusercontent.com/s/xt49a9hpzsbqr4l/train_ids.npy?dl=0'
valid_ids_url = 'https://dl.dropboxusercontent.com/s/v68xlv0fs5ewkqo/valid_ids.npy?dl=0'

# define file path
path = Path(__file__).parent
INPUT_FILE_SRC = path/'input.txt'
PREDICTION_FILE_SRC = path/'predictions.txt'
# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# helper func to download fastai learner model
async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


# helper func to setup fastai learner model
async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    await download_file(train_ids_url, path/'models'/ f'train_ids.npy')
    await download_file(valid_ids_url, path/'models'/f'valid_ids.npy')
    empty_data = TextClasDataBunch.load(path, 'models')
    learn = text_classifier_learner(empty_data)
    learn.load(model_file_name, with_opt=False)
    return learn

# Setup fastai learner
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


# API route
@app.route('/api', methods=['POST'])
def api():
    """API function
    All model-specific logic to be defined in the get_model_api()
    function
    """
    input = request.json
    app.logger.info("api_input: " + str(input))
    # output_data = model_api(input_data)
    # get predictions
    prediction, _, losses = learn.predict(str(input))
    output = "FOUL!!" if prediction.obj == '1' else "no foul"
    app.logger.info("api_output: " + str(output))
    data = {'input': input, 'output': output}
    response = jsonify(data)
    return response


@app.route('/')
def index():
    return render_template('b.html')
    # return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used to run locally.
    # app.run(debug=True)
    # This is used to deploy.
    app.run(host='0.0.0.0', port=8080, debug=True)
