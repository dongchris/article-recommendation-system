# Launch with
#
# gunicorn -D --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc


from flask import Flask, render_template
from doc2vec import *
import sys

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
  	
    return render_template('articles.html', test = articles)

@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    articlename = '%s/%s' % (topic, filename)
    content = ""
    title = ""
    for i in range(len(articles)):
    	if articles[i][0] == articlename:
    		content = re.sub('\n', '<br>',articles[i][2])
    		title = articles[i][1]

    similar = recommended(articlename, articles, 5)
    similararticle = [similar[i][1][1] for i in range(5)]
    simtitle = [similar[i][1][0] for i in range(5)]

    zipart = zip(similararticle, simtitle)

    return render_template('article.html', test = content, test2 = title, test3 = zipart)

# initialization
i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
articles = load_articles(articles_dirname, gloves)
