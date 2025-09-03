import requests

api_key = "THE API KEY WAS HERE"
client_id = "THE CLIENT ID WAS HERE"


def upload_to_imgur(image_path):
    headers = {'Authorization': f'Client-ID {client_id}'}
    files = {'image': open(image_path, 'rb')}
    url = "https://api.imgur.com/3/upload"

    response_imgur = requests.post(url, headers=headers, files=files)
    if response_imgur.status_code == 200:
        return response_imgur.json()['data']['link']
    else:
        print("Error uploading to Imgur:", response_imgur.status_code)
        return None


def recognised(local_image_path):
    imgur_url = upload_to_imgur(local_image_path)

    if imgur_url:
        base_url = "https://serpapi.com/search.json"

        params = {
            "engine": "google_lens",
            "url": imgur_url,
            "api_key": api_key,
        }

        api_url = f"{base_url}?{('&'.join([f'{key}={value}' for key, value in params.items()]))}"

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()

            titles = [item.get("title", "") for item in data.get("visual_matches", [])]
            contains_halsey = any("Halsey" in title for title in titles)



            if contains_halsey:
                return True
            else:
                return False

        else:
            print("Error:", response.status_code)
    else:
        print("Failed to upload to Imgur.")
