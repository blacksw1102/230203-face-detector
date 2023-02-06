import threading

from_number = 1
to_number = 100


def first_thread():
    print("First thread starting")
    for i in range(from_number, to_number):
        print("First thread counting ", i)


def second_thread():
    print("Second thread starting")
    for i in range(from_number, to_number):
        print("Second thread counting ", i)


if __name__ == "__main__":
    t1 = threading.Thread(target=first_thread)
    t2 = threading.Thread(target=second_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Both threads completed")
