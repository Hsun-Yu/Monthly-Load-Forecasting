# %%
import mlflow
import sys, getopt
from dotenv import load_dotenv
import os

def main(argv):
    experiment_id = ''
    outputfile = ''
    try:
      opts, args = getopt.getopt(argv,"ho:e:")
    except getopt.GetoptError:
      print('python download_csv.py -o <output_filename> -e <experiment_id>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('python download_csv.py -o <output_filename> -e <experiment_id>')
         sys.exit()
      elif opt in ("-e"):
         experiment_id = arg
      elif opt in ("-o"):
         outputfile = arg

    print('Experiment Id is', experiment_id)
    print('Output file is', outputfile)
    print('Downloading...')
    load_dotenv()
    tracking_url = os.getenv("mlflow_tracking_uri")
    mlflow.set_tracking_uri(tracking_url)
    mlflow.search_runs(experiment_ids=[int(experiment_id)]).to_csv(outputfile, index=False)
    print('Done!')

if __name__ == "__main__":
   main(sys.argv[1:])