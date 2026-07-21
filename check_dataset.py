import pandas as pd

df = pd.read_csv('data/consolidated_metadata.csv')
print(f'Total images: {len(df)}')
print(f'Zombies: {len(df[df.Class=="Zombie"])}')
print(f'Humans (Default): {len(df[df.Class=="Default"])}')
