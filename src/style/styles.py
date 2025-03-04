CSS_STYLE = """
<style>
    body {
        background-color: #FAF3E0 !important;
        color: #FAF3E0;
    }
    #chatbox {
        background-color: #0072BB !important;
        color: black !important;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    #header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
        background-color: #0072BB;
        border-radius: 10px;
    }
    #header h1 {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    #header p {
        color: white;
        font-size: 14px;
        margin: 0;
    }
    .gradio-container {
        background-color: #FAF3E0 !important;
    }
    #logo-container {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }
    #logo {
        max-width: 172px;
        max-height: 50px;
        width: auto;
        height: auto;
        object-fit: contain;
        display: block;
        cursor: pointer;
    }
    #footer {
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        background-color: #FAF3E0;
        border-top: 2px solid #0072BB;
        margin-top: 20px;
    }
    #footer a {
        color: #0072BB;
        font-weight: bold;
        text-decoration: none;
    }
    /* 🔹 Input Field Styling */
    input[type="text"] {
        background: white !important;
        color: black !important;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        border: 1px solid #ccc;
    }
    input[type="text"]:focus {
        border-color: #0072BB !important;
        outline: none;
        box-shadow: 0 0 8px rgba(0, 114, 187, 0.3);
    }
</style>
"""
