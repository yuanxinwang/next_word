# load Flask
import flask
#import torch
#from transformers import GPT2Tokenizer
import pickle
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

app = flask.Flask(__name__)
@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}
    # get the request parameters
    params = flask.request.json
    if (params == None):
        params = flask.request.args
    # if parameters are found, echo the msg parameter

    ''' Not support for GCP
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = torch.load("static/gpt2")
    indexed_tokens = tokenizer.encode(params.get("msg"))
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    '''
    try:
        try:
            module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
            embed = hub.KerasLayer(module_url)
        except:
            data["response"] = "fail to load emb"
            data["success"] = False
            return flask.jsonify(data)
        try:
            with open("static/tokenizer.pickle", "rb") as handle:
                tokenizer = pickle.load(handle)
            model = load_model("static/my_model.h5")
        except:
            data["response"] = "fail to load tokenizer"
            data["success"] = False
            return flask.jsonify(data)

        def predict(model, sentence):
            predicted = model.predict_classes(embed([sentence]), verbose=0)
            for word, index in tokenizer.word_index.items():
                if index == predicted[0]:
                    output_word = word
                    return (output_word)
                    break
            return null
        data["response"] = predict(model, params.get("msg"))#tokenizer.decode([predicted_index])
        data["success"] = True
    except:
        data["response"] = "fail to predict"
        data["success"] = False
    # return a response in json format
    return flask.jsonify(data)


# start the flask app, allow remote connections
if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)