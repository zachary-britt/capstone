''' Lifted straight from Galvanize DSI flask solutions '''


from flask import Flask, render_template, request, jsonify
import spacy
import spacy_textcat



app = Flask(__name__)

model = spacy_textcat.main(data_name = '', model_name='spacy-22.Tr7m5',
        model_getter=True, tanh_setup=True )




@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('form/index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('form/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['article_body'])
    pred = str(model.single_query(data))
    return render_template('form/predict.html', article=data, predicted=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
