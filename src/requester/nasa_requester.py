import earthaccess

def nasa():
    print("starting nasa")

    earthaccess.login(strategy="environment", persist=True)

    results = earthaccess.search_data(
        short_name='MOD09GA',
        bounding_box=(-10, 20, 10, 50),
        temporal=("2018-02", "2019-03"),
        count=10
    )
    files = earthaccess.download(results, f"C:/Bachelorarbeit/nasa")


if __name__ == "__main__":
    nasa()