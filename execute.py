from flask import Flask,render_template,request,url_for,redirect
import driver

app=Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/output')
def out():
    return render_template("op.html")


@app.route('/',methods=['POST'])
def getvalue():
    global tick
    tick=request.form['symbol']
    global p
    p = driver.driver(tick)
    return render_template('op.html',output=p)


if __name__=="__main__":
    app.run(debug=True)