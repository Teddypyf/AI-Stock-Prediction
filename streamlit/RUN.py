import streamlit as st
import subprocess

def main():
    st.title("Select Script to Run")

    # 用户选择要运行的脚本
    script_to_run = st.selectbox("Select Script", ["Get_Data.py","AI.py","Perdiction.py","Plot.py"])

    if st.button("Run Script"):
        # 在 Streamlit 应用程序中运行选定的脚本
        subprocess.run(["streamlit", "run", script_to_run])

if __name__ == "__main__":
    main()
