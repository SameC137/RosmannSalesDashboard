import pandas as pd
import pandas_profiling
import streamlit as st

# from streamlit_gallery.utils import readme
from streamlit_pandas_profiling import st_profile_report


def main():
    dataset = "train.csv"

    df = pd.read_csv(dataset)
    pr = gen_profile_report(df, explorative=True)

    st.write(f"🔗 [Rossman Sales]({dataset})")
    st.write(df)

    with st.beta_expander("REPORT", expanded=True):
        st_profile_report(pr)
    pr.to_file('train.html')


@st.cache(allow_output_mutation=True)
def gen_profile_report(df, *report_args, **report_kwargs):
    return df.profile_report(*report_args, **report_kwargs)


if __name__ == "__main__":
    main()