# Summarizer Studio (Streamlit)

This project hosts a ChatGPT-style summarizer dashboard in Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy globally on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Create a Streamlit Community Cloud app.
3. Set:
   - Main file path: `app.py`
   - Python version: default
4. First run will download the model weights.

## Notes

- Default model is CPU-friendly. You can switch to a higher quality model in the sidebar.
- Long documents are auto-chunked.
