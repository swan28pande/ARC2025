import argparse
import tinker
import urllib.request
import os
import dotenv
dotenv.load_dotenv()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download model checkpoint from Tinker")
    parser.add_argument("--tinker_path", required=True, help="Unique model path in Tinker")
    parser.add_argument("--safedir", required=True, help="Local path to save directory")
    # parser.add_argument("--safe_file_name", required=True, help="Local file name to save checkpoint")
    args = parser.parse_args()

    # Create Tinker clients
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    # Build the Tinker path dynamically
    tinker_path = args.tinker_path
    # Request checkpoint archive URL
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    checkpoint_archive_url_response = future.result()

    #check if the safe dir exists, if not create it
    if not os.path.exists(args.safedir):
        os.makedirs(args.safedir)

    # Download the archive
    print(f"Downloading checkpoint from: {checkpoint_archive_url_response.url}")
    urllib.request.urlretrieve(checkpoint_archive_url_response.url, os.path.join(args.safedir, "archive.tar"))
    print(f"Checkpoint successfully downloaded to: {os.path.join(args.safedir, "archive.tar")}")
    print(f"URL expires at: {checkpoint_archive_url_response.expires}")

if __name__ == "__main__":
    main()