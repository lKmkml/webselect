import itertools
import os
from collections import defaultdict
from fileinput import filename

from flask import Flask, flash, redirect, render_template, request
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename

app=Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'homework/uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)
@app.route('/search_1', methods = ['GET', 'POST'])
def search_1():
    return render_template('search.html')

@app.route('/search_2', methods = ['GET', 'POST'])
def search_2():
    if request.method == 'POST':
        word = request.form['word']
        dir_path = r'homework/uploads/'
        count = 0
        # Iterate directory
        for path in os.listdir(dir_path):
        # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
    articles = []
    for i in range(count):
        f = open(f"homework/uploads/wiki_article_{i}.txt", "r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)
    dictionary = Dictionary(articles)
    #print(dictionary)
    word_id = dictionary.token2id.get(word)
    msg = ("Found : "+word+' word id = '+str(word_id))
    return render_template('search.html',msg = msg)

@app.route('/top5_1', methods = ['GET', 'POST'])
def top5_1():
    return render_template('top5.html')
@app.route('/top5_2', methods = ['GET', 'POST'])
def top5_2():
    if request.method == 'POST':
        dir_path = r'homework/uploads/'
        count = 0
        # Iterate directory
        for path in os.listdir(dir_path):
        # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
    articles = []
    for i in range(count):
        f = open(f"homework/uploads/wiki_article_{i}.txt", "r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)
    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
    doc = corpus[0]
    msgtopbow=[]
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count

    sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
    for word_id, word_count in sorted_word_count[0:5]:
        b=(dictionary.get(word_id), word_count)
        msgtopbow.append(b)
    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
    msgtop = []
    for term_id, weight in sorted_tfidf_weights[:5]:
        a = dictionary.get(term_id), weight
        msgtop.append(a)
    return render_template('top5.html', msgtop = msgtop,msgtopbow=msgtopbow)

if __name__ == "__main__":
    app.run(host = '127.0.0.1',port = 5000, debug = True)
