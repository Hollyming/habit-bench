from mirix.log import get_logger

logger = get_logger(__name__)


def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    if response.get("type") in [
        "agent_response_end",
        "agent_response_error",
        "command_response",
        "server_error",
    ]:
        return True
    else:
        return False


def print_server_response(response):
    """Turn response json into a nice print"""
    if response["type"] == "agent_response_start":
        logger.info("[agent.step start]")
    elif response["type"] == "agent_response_end":
        logger.info("[agent.step end]")
    elif response["type"] == "agent_response":
        msg = response["message"]
        if response["message_type"] == "internal_monologue":
            logger.info("[inner thoughts] %s", msg)
        elif response["message_type"] == "assistant_message":
            logger.info("%s", msg)
        elif response["message_type"] == "function_message":
            pass
        else:
            logger.info(response)
    else:
        logger.info(response)


def shorten_key_middle(key_string, chars_each_side=3):
    """
    Shortens a key string by showing a specified number of characters on each side and adding an ellipsis in the middle.

    Args:
    key_string (str): The key string to be shortened.
    chars_each_side (int): The number of characters to show on each side of the ellipsis.

    Returns:
    str: The shortened key string with an ellipsis in the middle.
    """
    if not key_string:
        return key_string
    key_length = len(key_string)
    if key_length <= 2 * chars_each_side:
        return "..."  # Return ellipsis if the key is too short
    else:
        return key_string[:chars_each_side] + "..." + key_string[-chars_each_side:]
