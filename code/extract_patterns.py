import pandas as pd
import re

# Assuming the content of the text file is stored in a string variable called `content`
with open("patterns.txt", "r") as file:
    content = file.read()

# Regular expression to extract the attributes from the XML tags
pattern = re.compile(r'<r p1="(.*?)" p2="(.*?)" p3="(.*?)" p4="(.*?)" p5="(.*?)" p6="(.*?)">(.*?)<\/r>')

matches = pattern.findall(content)

data = []
for match in matches:
    data.append({
        "p1": match[0],
        "p2": match[1],
        "p3": match[2],
        "p4": match[3],
        "p5": match[4],
        "p6": match[5],
        "r": match[6]
    })

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("output.csv", index=False)
