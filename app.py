from flask import Flask, render_template, request
from pre_process import df, hybrid_recomm

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['movie']
    results, error = hybrid_recomm(title)
    if error:
        return render_template('results.html', title=title, error=error)
    return render_template('results.html', title=title, results=results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
