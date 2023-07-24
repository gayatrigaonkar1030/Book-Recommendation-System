import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer




st.header("Book Recommender System")

final_data = pickle.load(open('C:\\Users\\HP\\PROJECT2\\deployment\\final_data.pkl','rb'))
pivot_tables = pickle.load(open('C:\\Users\\HP\\PROJECT2\\deployment\\pivot_tables.pkl','rb'))
book_names = pickle.load(open('C:\\Users\\HP\\PROJECT2\\deployment\\book_names.pkl','rb'))

def content_based_recommender(book_title):
    
    book_title = str(book_title)
    if book_title in final_data['title'].values:
        rating_counts = pd.DataFrame(final_data['title'].value_counts())
        rare_books = rating_counts[rating_counts['title'] <= 50].index
        common_books = final_data[~final_data['title'].isin(rare_books)]
        
        if book_title in rare_books:
            
            random = pd.Series(common_books['title'].unique()).sample(2).values
            print('There are no recommendations for this book')
            print('Try: \n')
            print('{}'.format(random[0]),'\n')
            print('{}'.format(random[1]),'\n')
        
        else:
            
            common_books = common_books.drop_duplicates(subset=['title'])
            common_books.reset_index(inplace= True)
            common_books['index'] = [i for i in range(common_books.shape[0])]
            target_cols = ['title','author','publisher']
            common_books['combined_features'] = [' '.join(common_books[target_cols].iloc[i,].values) for i in range(common_books[target_cols].shape[0])]
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(common_books['combined_features'])
            cosine_sim = cosine_similarity(count_matrix)
            index = common_books[common_books['title'] == book_title]['index'].values[0]
            sim_books = list(enumerate(cosine_sim[index]))
            sorted_sim_books = sorted(sim_books,key=lambda x:x[1],reverse=True)[1:6]
            #sorted_sim_books
            
            # Create a horizontal layout with columns
            columns = st.columns(len(sorted_sim_books))
            
            books = []
            for i in range(len(sorted_sim_books)):
                book_list = []
                temp_book = common_books[common_books['index'] == sorted_sim_books[i][0]]
                #temp_book
                
                book_list.extend(list(temp_book.drop_duplicates('title')['title'].values))
                book_list.extend(list(temp_book.drop_duplicates('title')['author'].values))
                book_list.extend(list(temp_book.drop_duplicates('title')['img_m'].values))
                books.append(book_list)
                #books    
                title =book_list[0]
                author =book_list[1]
                image_url = book_list[2]  # Access the correct index of data where image URLs are stored
                with columns[i]:
                    st.write("<h6>",title,"</h6>",unsafe_allow_html=True)
                    st.image(image_url, width=120)
                    st.write("Author : ",author)
                
                
                
selected_book = st.selectbox("Type or Select Book Name",book_names)

if st.button('Show Recommendation'):
     st.write("<h5>You Searched :",selected_book,"</h5>",unsafe_allow_html=True)
     st.write("<h5>The Recommended books are :</h5>",unsafe_allow_html=True)
     recommended_books = content_based_recommender(selected_book)
     
     
    