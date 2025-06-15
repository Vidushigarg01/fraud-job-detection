import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    categorical_fields = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    df[text_fields] = df[text_fields].fillna('')
    df[categorical_fields] = df[categorical_fields].fillna('Unknown')

    return df
