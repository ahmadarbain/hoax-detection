from flask import Flask, render_template, request
import marks as m

app = Flask(__name__, static_folder='./static')


@app.route("/", methods = ["GET","POST"])
def marks():
    text = ""
    predict = ""
    detect = ""
    if request.method == "POST":
        text = request.form["text"]
        mark_predict = m.marks_prediction(text)
        print(mark_predict)
        predict = mark_predict[0]*100 
        detect = mark_predict[1]


    return render_template(
        "index.html", 
        my_prediction = detect, 
        prediction =predict, 
        text=text
    )


@app.route('/')
def submit():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)