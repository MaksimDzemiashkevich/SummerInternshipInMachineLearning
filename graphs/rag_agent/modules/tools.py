from langchain.tools import tool


@tool
def convert_bar_to_mpa(value: float) -> float:
    """Convert a pressure from bar to megapascals (MPa). `value` is the pressure in bar. 1 bar = 0.1 MPa."""
    return round(value * 0.1, 2)


@tool
def convert_psi_to_mpa(value: float) -> float:
    """Convert a pressure from psi to megapascals (MPa). `value` is the pressure in psi. 1 psi = 0.006895 MPa."""
    return round(value * 0.006895, 2)


@tool
def convert_inch_to_mm(value: float) -> float:
    """Convert a length from inches to millimetres. `value` is the length in inches. 1 inch = 25.4 mm."""
    return round(value * 25.4, 2)


@tool
def convert_lb_to_kg(value: float) -> float:
    """Convert a weight from pounds to kilograms. `value` is the weight in pounds. 1 lb = 0.4536 kg."""
    return round(value * 0.4536, 2)


@tool
def convert_kelvin_to_celsius(value: float) -> float:
    """Convert a temperature from Kelvin to degrees Celsius. `value` is the temperature in Kelvin."""
    return round(value - 273.15, 2)


@tool
def convert_fahrenheit_to_celsius(value: float) -> float:
    """Convert a temperature from Fahrenheit to degrees Celsius. `value` is the temperature in Fahrenheit."""
    return round((value - 32) * 5 / 9, 2)


@tool
def remove_characters(text: str, characters: str) -> str:
    """Remove EVERY occurrence of the given characters from the text and return the result.

    Use this whenever a rule says to drop a separator or punctuation mark, instead of retyping
    the value: the removal is then applied to every occurrence, and nothing else changes.
    `characters` is the set of characters to drop, e.g. "-" or "-,".
    """
    return "".join(ch for ch in text if ch not in set(characters))


@tool
def replace_substring(text: str, find: str, replace_with: str) -> str:
    """Replace EVERY occurrence of `find` with `replace_with` in the text and return the result.

    Use this whenever a rule says to swap one separator, word or symbol for another (for example
    a hyphen for a tilde, or the word "and" for "&"), instead of retyping the value.
    """
    return text.replace(find, replace_with)


array_of_tools = [convert_bar_to_mpa, convert_psi_to_mpa, convert_inch_to_mm,
                  convert_lb_to_kg, convert_kelvin_to_celsius, convert_fahrenheit_to_celsius,
                  remove_characters, replace_substring]
