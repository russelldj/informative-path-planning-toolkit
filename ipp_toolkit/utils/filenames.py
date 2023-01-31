def format_string_with_iter(template: str, iter: int):
    try:
        formatted_str = template.format(iter)
        return formatted_str
    except:
        return None
