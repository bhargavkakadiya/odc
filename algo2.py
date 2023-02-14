# Second algorithm to predict
import numpy as np
import pandas as pd

try:
       from prophet.plot import plot_plotly, plot_components_plotly
       from prophet import Prophet
except:
       !pip install prophet
       from prophet.plot import plot_plotly, plot_components_plotly
       from prophet import Prophet

hours=['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']

def get_input(local=False):
    if local:
        print("Reading local file")
        return 'ocean.csv'
    dids = os.getenv("DIDS", None)
    if not dids:
        print("No DIDs found in environment. Aborting.")
        return
    dids = json.loads(dids)
    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")
        return filename

def run_prophet(local=False):
    filename = get_input(local)
    df = pd.read_csv(filename)
    df.columns = ['official_code', 'cabin_name', 'date', 'pollutant_code', 'pollutant', 'unit',
       'station_type', 'area_type', 'municipality_code', 'municipality', 'county_code',
       'county', '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h', 'alt', 'lat',
       'lon', 'geo']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y') # Convert to datetime
    # Impute missing values with previous and next values
    df = df.groupby(['cabin_name', 'pollutant_code']).apply(lambda group: group.interpolate(method='ffill', axis=1))
    df = df.groupby(['cabin_name', 'pollutant_code']).apply(lambda group: group.interpolate(method='bfill', axis=1))

       predictions = {}
       df_ = df.loc[df.pollutant == 'NO2'].groupby(['cabin_name'])
       for cabin in df_.groups.keys():
           # print(cabin, df_.get_group(cabin).date.min(), df_.get_group(cabin).date.max())
           # get days total till 28th feb 2023
           days_to_predict = (datetime.datetime(2023, 2, 28) - df_.get_group(cabin).date.max()).days
           dft = df_.get_group(cabin)[hours+['date']]
           predictions_hourly = {}
           for hour in hours:
               dfth = dft[['date', hour]]
               dfth.columns = ['ds', 'y']
               m = Prophet()
               m.fit(dfth)
               future = m.make_future_dataframe(periods=days_to_predict)
               forecast = m.predict(future)
               dff = forecast[['ds', 'yhat']].tail(days_to_predict)
               dff.columns = ['date', hour]
               # add to predictions
               predictions_hourly[hour] = dff

           predictions[cabin] = predictions_hourly


    filename = "model_algo2_result.pickle" if local else "/data/outputs/result"
    with open(filename, "wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(predictions, pickle_file)

if __name__ == "__main__":
    local = True  
    run_prophet(local)
