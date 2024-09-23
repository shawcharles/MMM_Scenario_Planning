mkdir -p .streamlit

echo "\
[general]\n\
email = \"nmaidanenko4@gmail.com\"\n\
" > ./.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = ${PORT:-8501}
enableCORS = false\n\
maxUploadSize = 200\n\
" >>./.streamlit/config.toml
