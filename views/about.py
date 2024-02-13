import streamlit as st


def init():
    st.title("About VisioCraft")

    st.markdown(
        """
        **VisioCraft** is an innovative image and video processing web application designed to elevate your visual content. 
        With state-of-the-art technology and a user-friendly interface, VisioCraft offers a seamless experience in enhancing and analyzing images and videos.

        ### Key Features:
        - **Data source:** Videos from various sources, including YouTube, Pexels, and more. data can be also uploaded from the local machine.
        - **Image Mode:** Upload an image and explore a variety of image processing filters, including Grayscale conversion, Histogram Equalization, Gaussian Blur, and many more.
        - **Video Mode:** Analyze videos with the ability to stop processing at any time.
        - **Webcam Mode:** Experience real-time emotion and gender analysis using your webcam.
        - **Metrics and Insights:** Dive into detailed metrics that provide valuable insights into the analyzed content.
        - **Filters and Enhancements:** Choose from a diverse set of filters, including Laplacian, Sobel, Sharpen, Enhance RGB quality, Remove Blur effect, Median filter, and more.
        - **Saving processed content:** Save processed images and videos to **HDFS**.
        - **Clean dashboard:** Show  the details of the processed content and the metrics of the processed content.
        - **Browse processed data:** Browse the processed data icluding recent images..etc.
        - **Modern UI:** A sleek and intuitive user interface that makes image and video processing a breeze.

        ### Metrics and Insights:

        Dive into detailed metrics that provide valuable insights into the analyzed content. Track the number of men, women, and overall student satisfaction based on facial expressions and gender detection.

        ### Filters and Enhancements:

        Choose from a diverse set of filters, including Laplacian, Sobel, Sharpen, Enhance RGB quality, Remove Blur effect, Median filter, and more. Customize your image processing experience by selecting and combining filters.

        ### Contributors:
        The **VisioCraft** application was developed by a team of future **data scientists** and **software engineers**. Meet the team behind the app 😃:

        | **Name**              | **GitHub Profile**                                                     |
        |-----------------------|------------------------------------------------------------------------|
        | **LAGHJAJ Abdellatif**| [GitHub - Laghjaj Abdellatif](https://github.com/abdellatif-laghjaj)   |
        | **ADARDOUR Naima**    | [GitHub - Adardour Naima](https://github.com/naima-adardor)            |
        | **BOUCHHAR Maryam**   | [GitHub - Bouchhar Maryam](https://github.com/MaryamBouchhar)          |

        ### About Us:

        We are a team passionate about leveraging cutting-edge technologies to create intuitive and impactful applications. Our goal is to make advanced image and video processing accessible to everyone, whether you're an enthusiast or a professional.

        Feel free to explore, experiment, and enjoy the visual transformations powered by our app. Thank you for being a part of this exciting journey!
        """
    )
