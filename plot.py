import config
from binance.client import Client
from datetime import datetime,timedelta
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.text import MIMEText
import time


def convert_timestamp_to_utc(timestamp_in_milliseconds):
    timestamp_in_seconds = timestamp_in_milliseconds / 1000.0
    return datetime.utcfromtimestamp(timestamp_in_seconds)


def get_pnl(income_history,yesterday = 1):
    
    aggregations = {
    'symbol': lambda x: mode(x).mode[0] if mode(x).count[0] > 1 else x.iloc[0],
    'income': 'sum', 
    'date': lambda x: mode(x).mode[0]  
}
    
    
    df = pd.DataFrame(income_history)
    df['utc_time'] = df['time'].apply(convert_timestamp_to_utc)
    df['date'] = df['utc_time'].dt.day
    df['income'] = df['income'].astype(float)
    df_commission = df[(df['incomeType']!='REALIZED_PNL' ) & (df['incomeType']!= 'TRANSFER')]
    df_PNL = df[df['incomeType']=='REALIZED_PNL']
    
    PNL = df_PNL.groupby('utc_time').agg(aggregations).reset_index()
    if yesterday == 1:
        yesterday = datetime.utcnow().day - 1
    else:
        yesterday = datetime.utcnow().day
    
    yesterday_df = PNL[PNL['date']== yesterday]
    df_commission_yesterday = df_commission[df_commission['date']==yesterday]
    
    income = yesterday_df['income'].sum()
    commision = df_commission_yesterday['income'].sum()
    
    total_yesterday = income + commision
    
    print(f'income : {income} , commision : {commision}')
    
    return total_yesterday

def plot_day_over_day(df):

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)

    df['DateLabel'] = df['Date'].dt.strftime('%d-%m')

    # Use a bar plot and use color to differentiate positive and negative values
    bars = ax.bar(df['DateLabel'], df['Percentage Change'], color=[
                  'g' if x >= 0 else 'r' for x in df['Percentage Change']])

    # Rotate x-axis labels for better visibility
    plt.xticks(df['DateLabel'], rotation=90,
               fontsize=12, weight='bold', color='black')

    # Set y-ticks properties
    ax.tick_params(axis='y', colors='black', labelsize=12)

    # Display data labels
    for bar, date in zip(bars, df['Date']):
        yval = bar.get_height()
        if not np.isnan(yval):  # Check if yval is not NaN
            if yval >= 0:
                label_position = yval + 0.01
            else:
                label_position = yval - 0.01
            ax.text(bar.get_x() + bar.get_width()/2., label_position,
                    f"{yval:.2f}%\n{date.strftime('%d-%m')}", ha='center', va='bottom', rotation=0, fontsize=10, weight='bold')

    plt.title("Percentage Change", fontsize=16, weight='bold')
    plt.ylabel("Percentage Change (%)", fontsize=14, weight='bold')
    plt.xlabel("Date", fontsize=14, weight='bold')

    # Find the most common month
    most_common_month = df['Date'].dt.strftime('%B %Y').mode()[0]

    # Display the most common month on the plot
    plt.text(0.99, 0.85, most_common_month, transform=ax.transAxes,
             fontsize=14, weight='bold', ha='right')

    # Adjust layout to ensure labels are not cut off
    fig.tight_layout()

    # Save the plot to disk
    plt.savefig("daily_change.png", bbox_inches='tight')

    plt.show()

def send_mail(filename, subject='SARAVANA BHAVA'):
    from_ = 'gannamanenilakshmi1978@gmail.com'
    to = 'vamsikrishnagannamaneni@gmail.com'

    message = MIMEMultipart()
    message['From'] = from_
    message['To'] = to
    message['Subject'] = subject
    body_email = 'SARAVANA BHAVA !'

    message.attach(MIMEText(body_email, 'plain'))

    attachment = open(filename, 'rb')

    x = MIMEBase('application', 'octet-stream')
    x.set_payload((attachment).read())
    encoders.encode_base64(x)

    x.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    message.attach(x)

    s_e = smtplib.SMTP('smtp.gmail.com', 587)
    s_e.starttls()

    s_e.login(from_, 'upsprgwjgtxdbwki')
    text = message.as_string()
    s_e.sendmail(from_, to, text)
    print(f'Sent {filename}')


initial_capital = 385
client=Client(config.api_key,config.secret_key)

current_time = datetime.utcnow()
if current_time.hour == 0:
    daily_PNL = pd.read_csv('daily_pnl.csv')
    income_history = client.futures_income_history(limit = 500)
    now = datetime.utcnow() - timedelta(days=1)
    daily_PNL = daily_PNL.append({'Date': now.strftime('%d-%m-%Y'), 'PNL': get_pnl(income_history)}, ignore_index=True)
    daily_PNL['Percentage Change'] = (daily_PNL['PNL']/initial_capital) * 100
    daily_PNL.drop_duplicates(subset=['Date'],inplace=True)
    daily_PNL.to_csv('daily_pnl.csv',index = False)
    plot_day_over_day(daily_PNL)
    send_mail("daily_change.png")
    time.sleep(3700)
else:
    print(f'Sleeping for 20 minutes :{current_time}')
    time.sleep(1200)