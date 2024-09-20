


import os
import subprocess
import argparse
import openai
import requests


def get_input(prompt, default=None):
    return default if default else input(prompt)

# Get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--symbol', type=str, help='Symbol')
parser.add_argument('--start_date', type=str, help='Start Date')
parser.add_argument('--end_date', type=str, help='End Date')
args = parser.parse_args()

# Define default values for input (if needed)
symbol = get_input("Enter the symbol: ", default=args.symbol)
start_date = get_input("Enter the start date: ", default=args.start_date)
end_date = get_input("Enter the end date: ", default=args.end_date)

# List of prediction scripts
prediction_scripts = [

    'pred_price.py',
    'pred_vol.py',
    'pred_macd.py',
    'pred_rsi.py',
    'pred_bbl.py',
    'pred_bbm.py',
    'pred_bbu.py',
    'pred_sma.py',
    'pred_ema.py',
    'pred_stochk.py',
    'pred_stochd.py',
    'pred_adx.py',
    'pred_will.py',
    'pred_cmf.py',
    'pred_psari.py',
    'pred_psars.py',
    'ta.py',
    'sen.py',
    'prompt.py',
]

# Run each prediction script with the provided input
for script in prediction_scripts:
    script_path = os.path.join(os.path.dirname(__file__), script)
    command = f"python {script_path} --symbol {symbol} --start_date {start_date} --end_date {end_date}"
    subprocess.run(command, shell=True)



def hugging_chat_api(prompt):
    url = "https://api-inference.huggingface.co/models/facebook/opt-12"
    headers = {"Authorization": f"Bearer {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}"}
    data = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=data)
    return response.json()[0]['generated_text']


prompt = "Hello, how are you?"
response = hugging_chat_api(prompt)
print(response)

