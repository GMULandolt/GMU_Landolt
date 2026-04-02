import pandas as pd
    
data = {'col1': ['apple', 'banana', 'cherry']}
df = pd.DataFrame(data)
    
df['substring'] = df['col1'].str.slice(0, 3)  # Extracts the first 3 characters
print(df)