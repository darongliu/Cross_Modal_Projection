from apiclient.discovery import build

service = build("customsearch", "v1",
               developerKey="AIzaSyAjiNb5uQCCxqPe3vFl1ZRVZcxiGF7cvOk")

res = service.cse().list(
    q='cat play basketball',
    cx='000109622348442705360:f_fblmv4vyu',
    searchType='image',
    num=10,
    imgType='clipart',
    safe= 'off',
    start=21
).execute()

