import pandas as pd

# Load both datasets
hindi_df = pd.read_csv('C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\sms_filter\\hindi_balanced.csv')
spam_ham_df = pd.read_csv('C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\sms_filter\\spam_ham_india.csv')

# Standardize column names to ensure they match
hindi_df.columns = ['Msg', 'label']
spam_ham_df.columns = ['Msg', 'label']

# Add a language column
hindi_df['language'] = 'hindi'  # Add language 'hindi' to Hindi dataset
spam_ham_df['language'] = 'english'  # Add language 'english' to spam-ham dataset

# Combine both datasets
combined_df = pd.concat([spam_ham_df,hindi_df], ignore_index=True)

# Optionally, save the combined dataset to a new CSV file
combined_df.to_csv('combined_dataset.csv', index=False)



# # Load Hindi dataset (adjust file name/path and encoding if needed)
# df_hindi = pd.read_csv("hindi_dataset.csv")

# # Standardize column names (assuming columns are 'Msg' and 'Label')
# df_hindi.columns = ['Msg', 'label']
# df_hindi['label'] = df_hindi['label'].str.lower().str.strip()

# # Print original distribution
# print("Hindi dataset original distribution:")
# print(df_hindi['label'].value_counts())

# # Separate ham and spam rows
# df_hindi_ham = df_hindi[df_hindi['label'] == 'ham']
# df_hindi_spam = df_hindi[df_hindi['label'] == 'spam']

# # Count spam messages in Hindi dataset (should be around 747)
# spam_count_hindi = df_hindi_spam.shape[0]
# print("\nHindi dataset spam count:", spam_count_hindi)

# # Undersample ham messages to match spam count
# df_hindi_ham_sampled = df_hindi_ham.sample(n=spam_count_hindi, random_state=42)

# # Combine sampled ham with spam to create a balanced Hindi dataset
# df_hindi_balanced = pd.concat([df_hindi_ham_sampled, df_hindi_spam], ignore_index=True)
# print("\nBalanced Hindi dataset distribution:")
# print(df_hindi_balanced['label'].value_counts())

# #  save the balanced Hindi dataset
# df_hindi_balanced.to_csv("hindi_balanced.csv", index=False)
