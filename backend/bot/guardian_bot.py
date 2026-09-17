from .filters import classify_message


def process_message(text):
    message_type = classify_message(text)

    if message_type == "empty":
        return {
            "status": "ignored",
            "reason": "Empty message"
        }

    return {
        "status": "received",
        "input_type": message_type,
        "text": text.strip()
    }


if __name__ == "__main__":
    test_message = "This is a test message."

    print(process_message(test_message))