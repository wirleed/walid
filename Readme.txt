README.txt

This project contains two Python files:

1. Group1.py – The main Streamlit web application.
2. QR.py – A script to generate a QR code linking to the app.

====================
How to Run the App
====================

1. Make sure Python is installed on your system.

2. Install all required packages by running:

   pip install -r requirements.txt

3. Make sure the file "logo.png" is in the same folder as Group1.py. 
   The app will not display properly without it.

4. To launch the Streamlit app, run:

   streamlit run Group1.py

   This will open the app in your web browser.

============================
How to Generate the QR Code
============================

1. To generate the QR code image that links to the Streamlit app, run:

   python QR.py

2. This will create a file called "streamlit_qr.png" in the current folder.

You can scan this QR code with your phone to open the app.

====================
You're All Set!
====================
