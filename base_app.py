"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Data Visualization", "Meet The Team", "About Us"]
	#selection = st.sidebar.selectbox("Choose Option", options)

	with st.sidebar:
		selection = st.radio("Explore Our Options", options)
	

	# Building out the "Meet The Team" page
	if selection == "Meet The Team":
		st.info("About The Team")
		# You can read a markdown file from supporting resources folder
		st.markdown("""
		
		Our team consists of 5 talented data scientists and developers from various parts of Africa. These are:
		- Lungisa Joctrey
		- Christian Miri
		- Precious Orekha
		- Ibrahim Isarki
		- Caleb Tanko

		""")

# Building out the "About Us" page
	if selection == "About Us":
		st.write("""### PLICC Analytics""")
		#Company logo

		image = Image.open('resources/imgs/logo_thingy.jpeg')
		st.image(image, caption='')
		# You can read a markdown file from supporting resources folder
		st.markdown("""
		PLICC ANALYTICS specializes in Information Technology Services. We take 
		data and arrange it in such a way that it makes sense for business and individual users. We also build and train models that are capable of solving a wide range of classification problems. 
		
		Our team of leading data scientists work tirelessly to make your life and the life of your customers easy.
		"""
		)

		

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""
		The end goal of this research is to look at the tweets from individuals and determine if that particular person believes in the climate change or not. We have created and trained several models that can do this task.  

		Below is the data used to train the model.
		
		""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the prediction page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)


			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
