import os
from dotenv import load_dotenv
from google import genai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide"
)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

st.title("AI Data Analysis Assistant")

st.write("An interactive tool to explore your data, "
         "discover patterns, visualize results, "
         "and generate AI-powered insights.")
st.divider()

st.write("Upload a CSV file and let's analyze it!")

st.sidebar.title("📊 AI Data Analysis Assistant")

st.sidebar.write(
    "Upload a CSV file and explore "
    "your dataset using statistics, "
    "visualizations, and AI."
)

st.sidebar.info(
    "💡 Tip: Upload a clean CSV dataset "
    "for the best analysis results."
)
st.subheader("Upload Your Dataset")

st.write("Upload a CSV file to begin exploring "
         "your data and generating insights."
         )
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(
        f"Successfully uploaded: {uploaded_file.name}"
    )

    st.subheader("Dataset Preview")

    preview_rows = st.slider(
        "Number of rows to preview",
        min_value=5,
        max_value=min(50, len(df)),
        value=min(10, len(df))
    )

    st.caption(f"Showing {min(preview_rows, len(df))} rows"
               f"out of {len(df)} total rows."
               )

    st.dataframe(
        df.head(preview_rows)
    )

    st.subheader("Dataset Information")
    st.divider()
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.info(
        f"This dataset contains {df.shape[0]} rows "
        f"and {df.shape[1]} columns."
    )

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_info = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
                df.isnull().sum().values / len(df) * 100
        ).round(2)
    })

    st.dataframe(
        missing_info,
        use_container_width=True
    )

    st.subheader("Data Quality Status")

    total_missing = df.isnull().sum().sum()
    total_duplicates = df.duplicated().sum()

    if total_missing == 0 and total_duplicates == 0:

        st.success(
            "✅ The dataset has no missing values "
            "or duplicate rows."
        )

    else:

        if total_missing > 0:
            st.warning(
                f"⚠️ The dataset contains "
                f"{total_missing} missing values."
            )

        if total_duplicates > 0:
            st.warning(
                f"⚠️ The dataset contains "
                f"{total_duplicates} duplicate rows."
            )

    st.subheader("Low-Information Columns")

    low_information = []

    for column in df.columns:

        unique_count = df[column].nunique()

        if unique_count <= 1:
            low_information.append({
                "Column": column,
                "Unique Values": unique_count
            })

    if low_information:

        low_info_df = pd.DataFrame(low_information)

        st.dataframe(
            low_info_df,
            use_container_width=True
        )

    else:

        st.success(
            "✅ No low-information columns found."
        )


    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Data Visualization")

    numeric_columns = df.select_dtypes(include="number").columns

    categorical_columns = df.select_dtypes(include="object").columns

    if len(numeric_columns) > 0:
        selected_column = st.selectbox("Choose a numeric column", numeric_columns)
        st.markdown(f"**Distribution of {selected_column}**")
        fig, ax = plt.subplots()

        ax.hist(df[selected_column].dropna(), bins=20)
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {selected_column}")
        st.pyplot(fig)

        st.markdown(f"**Box Plot of {selected_column}**")

        fig, ax = plt.subplots()

        ax.boxplot(df[selected_column].dropna())

        ax.set_ylabel(selected_column)
        ax.set_title(f"Box Plot of {selected_column}")

        st.pyplot(fig)

    else:
        st.warning("No numeric columns found in this dataset.")

    st.subheader("Numerical Insights")

    if len(numeric_columns) > 0:
        numerical_insights = pd.DataFrame({
            "Column": numeric_columns,
            "Average": df[numeric_columns].mean().round(2).values,
            "Minimum": df[numeric_columns].min().values,
            "Maximum": df[numeric_columns].max().values
        })

        st.dataframe(
            numerical_insights,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No numerical columns are available for analysis.")

    st.subheader("Category Analysis")

    if len(categorical_columns) > 0:
        category_info = []

        for column in categorical_columns:
            counts = df[column].value_counts()

            if not counts.empty:
                category_info.append({
                    "Column": column,
                    "Unique Categories": df[column].nunique(),
                    "Most Common Category": str(counts.index[0]),
                    "Frequency": int(counts.iloc[0])
                })

        if category_info:
            st.dataframe(
                pd.DataFrame(category_info),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No categorical columns found.")

    st.subheader("Possible Outliers")

    outlier_info = []

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outlier_count = (
                (df[column] < lower_limit) |
                (df[column] > upper_limit)
        ).sum()

        outlier_info.append({
            "Column": column,
            "Possible Outliers": outlier_count,
            "Lower Limit": round(lower_limit, 2),
            "Upper Limit": round(upper_limit, 2)
        })

    if outlier_info:

        outlier_df = pd.DataFrame(outlier_info)

        st.dataframe(
            outlier_df,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns available "
            "for outlier analysis."
        )

    st.markdown("**Correlation Matrix**")

    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr()
        st.dataframe(correlation_matrix)
    else:
        st.info("At least 2 numeric columns are required.")

    st.markdown("**Correlation Heatmap**")

    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr()

        fig, ax = plt.subplots()
        ax.imshow(correlation_matrix)

        ax.set_xticks(range(len(correlation_matrix.columns)))

        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_yticklabels(correlation_matrix.columns)
        fig.colorbar(ax.images[0], ax=ax)
        st.pyplot(fig)
    else:
        st.info("At least 2 numeric columns are required.")

    st.markdown("**Compare Two Columns**")

    if len(numeric_columns) >= 2:
        column1 = st.selectbox("Choose a first column", numeric_columns)
        column2 = st.selectbox("Choose a second column", numeric_columns)

        correlation = df[column1].corr(df[column2])

        st.write(f"Correlation between {column1} and {column2}: " f"{correlation:.2f}")

    else:
        st.info("At least 2 numeric columns are required.")

    st.markdown("**Strongest Correlation**")
    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr()
        strongest_pair = None
        strongest_value = 0

        for column1 in numeric_columns:
            for column2 in numeric_columns:
                if column1 != column2:
                    value = abs(correlation_matrix.loc[column1, column2])
                    if value > strongest_value:
                        strongest_value = value
                        strongest_pair = (column1, column2)

        st.write(
            f"Strongest relationship: {strongest_pair[0]} and "f"{strongest_pair[1]}({correlation_matrix.loc[strongest_pair[0], strongest_pair[1]]:.2f})")
    else:
        st.info("At least 2 numeric columns are required.")

    st.markdown("**Relationship Between Columns**")

    if len(numeric_columns) >= 2:
        x_column = st.selectbox("Choose X-axis", numeric_columns, key="scatter_x")

        y_column = st.selectbox("Choose Y-axis", numeric_columns, key="scatter_y")

        fig, ax = plt.subplots()
        ax.scatter(df[x_column], df[y_column])

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{x_column} vs {y_column}")
        st.pyplot(fig)
        st.info(
            f"This chart shows the relationship between "
            f"{x_column} and {y_column}."
        )

    else:
        st.info("At least two numerical columns are needed for a scatter plot.")

    st.subheader("Data Cleaning Suggestions")
    for column in df.columns:
        missing_values = df[column].isnull().sum()
        if missing_values > 0:
            st.write(f"Consider handling{missing_values} missing values " f"in '{column}' .")

        if df[column].duplicated().sum() > 0:
            st.write(f"'{column}' contains repeated values.")


    st.markdown("**Category Visualization**")

    if len(categorical_columns) > 0:
        selected_category = st.selectbox("Choose a category", categorical_columns, key="categorical_chart")
        category_counts = df[selected_category].value_counts().head(10)

        fig, ax = plt.subplots()

        ax.bar(category_counts.index.astype(str), category_counts.values)
        ax.set_xlabel(selected_category)
        ax.set_ylabel(selected_category)
        ax.set_title(f"Top 10 {selected_category} Categories")

        plt.xticks(rotation=45)

        st.pyplot(fig)

    else:

        st.info("No categorical columns found.")

    st.header("Download Results")
    st.subheader("Download Dataset")
    csv_data = df.to_csv(index=False)
    st.download_button(label="Download Dataset as CSV", data=csv_data, file_name="analyzed_dataset.csv",
                       mime="text/csv")

    st.subheader("Download Analysis Report")

    report = "DATA ANALYSIS REPORT\n\n"

    report += f"Rows: {df.shape[0]}\n"
    report += f"Columns: {df.shape[1]}\n"
    report += f"Missing Values: {df.isnull().sum()}\n"
    report += f"Duplicate Rows: {df.duplicated().sum()}\n\n"

    report += "NUMERICAL COLUMNS\n"

    for column in numeric_columns:
        average = df[column].mean()
        minimum = df[column].min()
        maximum = df[column].max()

        report += (f"{column}: Average = {average:.2f}, " f"Minimum = {minimum}, Maximum = {maximum}\n")

    st.download_button("Download Analysis Report", report, "analysis_report.txt", "text/plain")


    st.header("AI Assistant")
    st.subheader("Ask AI AboutYour Dataset")

    st.write("Ask questions about your uploaded dataset, " "and AI will help you understand it.")

    question = st.text_input("Ask a question about your dataset",
                             placeholder="Example: Which columns have the most missing value?")
    st.caption(
        "You can ask about trends, missing values, "
        "numerical patterns, or relationships between columns."
    )

    if st.button("Ask AI"):
        if question.strip() == "":
            st.warning("Please enter a question about your dataset.")
        else:
            response = client.models.generate_content(model="gemini-3.6-flash", contents=f""" 
        You are a data analysis assistant.
        Analyze the following dataset information:

        Rows: {df.shape[0]}

        Columns: {df.shape[1]}

        Numerical columns: 
        {list(numeric_columns)}

        Categorical columns: 
        {list(categorical_columns)}

        Statistical summary:
        {df.describe().to_string()}

        Sample rows:
        {df.head().to_string()}

        Missing values:
        {df.isnull().sum().to_string()}

        Duplicate rows:
        {df.duplicated().sum()}

        Correlation matrix:
        {df[numeric_columns].corr().to_string() if len(numeric_columns) >= 2 else "Not enough numerical columns"}

        Possible outliners:
        {df[numeric_columns].apply(lambda column: (column < column.quantile(0.25) - 1.5 * (column.quantile(0.75) - column.quantile(0.25))) | (column > column.quantile(0.75) + 1.5 * (column.quantile(0.75) - column.quantile(0.25)))).sum().to_string()}
        User question: 
        {question}

        Answer based only on the dataset information provided above.Explain the answer clearly and simply.
        """)
            st.write(response.text)
            st.caption("AI-generated answer based on your uploaded dataset.")

    st.subheader("AI-Generated Insights")
    if st.button("Generate Insights"):
        response = client.models.generate_content(model="gemini-3.6-flash", contents=f"""
            You are a data analysis assistant.
            Analyze this dataset and provide the most important insights.

            Rows: {df.shape[0]}
            Columns:{df.shape[1]}

            Numerical columns:
            {list(numeric_columns)}

            Categorical columns:
            {list(categorical_columns)}

            Statistical summary:
            {df.isnull().sum().to_string()}

            Missing values:
            {df.isnull().sum().to_string()}

            Duplicate rows:
            {df.duplicated().sum()}

            Sample rows:
            {df.head().to_string()}

            Give 3 to 5 important insights.
            Keep the explanation clear and easy to understand.
            Do not invent information that is not supported by the dataset.
            """)
        st.write(response.text)
        st.caption("These insights are generated by AI based on the uploaded dataset.")

    st.subheader("AI Correlation Explanation")
    if st.button("Correlation Explanation"):
        if len(numeric_columns) >= 2:
            response = client.models.generate_content(model="gemini-3.6-flash", contents=f""" You are a data analysis assistant.
            Explain the relationships between the numerical columns in this dataset.

            Correlation matrix:
            {df[numeric_columns].corr().to_string()}

            Identify the strongest relationships and explain them in a simple language.

            Important:
            Correlation does not necessarily mean causation.
            Do not claim that one variable causes another.
            """)
            st.write(response.text)

        else:
            st.info(
                "At least two numerical columns are needed " "to analyze correlations."
            )


