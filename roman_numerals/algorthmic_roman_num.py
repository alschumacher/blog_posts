

class ArabicNumeralConverter:

    def __init__(self):

        # All we need to get started is what the value of a Roman numeral is
        self.primitives = {
            1: 'I',
            5: 'V',
            10: 'X',
            50: 'L',
            100: 'C',
            500: 'D',
            1000: 'M',
            1001: ''  # Convenient way to break loop application later on
        }
        self.primitives_as_ints = list(self.primitives.keys())

    def __call__(self, arabic_numeral):

        assert type(arabic_numeral) == str

        # Separates a numeral into what portion will be vinculum (if necessary) and what will be numeral
        vinculum_value, number_value = self._allocate_value_to_numeral_vinculum(arabic_numeral)

        vinculum = ''
        if vinculum_value:
            # Make vinculum portion
            vinculum = self._seq2roman_numeral(vinculum_value)
            vinculum = vinculum.lower()  # Using lowercase instead of overlines for convenience

        # Make numeral portion
        numeral = self._seq2roman_numeral(number_value)

        # Concatenate vinculum with numeral
        roman_numeral = vinculum + numeral

        return roman_numeral

    def _allocate_value_to_numeral_vinculum(self, arabic_numeral):

        # Default case
        vinculum = ''
        numeral = arabic_numeral

        # Vinculums start being used at 4000
        if int(arabic_numeral) > 3999 and int(arabic_numeral[-4:]) < 4000:
            numeral = arabic_numeral[-4:]
            vinculum = arabic_numeral[:-4] + '0'
        elif int(arabic_numeral) > 3999:
            numeral = arabic_numeral[-3:]
            vinculum = arabic_numeral[:-3]

        return vinculum, numeral

    def _make_digits(self, numeral):

        digits = []
        for i in range(len(numeral)):

            # Makes the digit value for some digit
            column_value = int(numeral[i] + ('0' * (len(numeral) - i - 1)))

            # If there's nothing in that column, don't add it.
            if column_value:
                digits.append(column_value)

        return digits

    def _lookup_symbol_by_digit(self, digit):
        return self.primitives[self.primitives_as_ints[digit]]

    def _exact_symbol_match_value(self, num_symbols, digit):

        if num_symbols in self.primitives_as_ints:
            num_symbols = 1
        else:
            digit /= num_symbols

        string = num_symbols * self.primitives[digit]

        return string

    def _get_previous_symbol(self, digit, i):
        previous_symbol = self._lookup_symbol_by_digit(i - 1)
        if '5' == str(self.primitives_as_ints[i - 1])[0] or self.primitives_as_ints[i - 1] > digit:
            previous_symbol = self._lookup_symbol_by_digit(i)
        return previous_symbol

    def _find_symbol_and_number(self, num_symbols, digit, i):
        stop_loop = False

        # The rule of 3 here
        if num_symbols <= 3:
            symbol_number = num_symbols * self._lookup_symbol_by_digit(i)
            stop_loop = True

        # For accomodating the special cases at 4 & 9
        elif digit == (self.primitives_as_ints[i + 1] - digit / num_symbols):
            previous_symbol = self._get_previous_symbol(digit, i)
            symbol_number = previous_symbol + self._lookup_symbol_by_digit(i + 1)
            stop_loop = True

        # Rule of subtraction here
        elif num_symbols > 5:
            symbol_number = self._lookup_symbol_by_digit(i) + (num_symbols - 5) * self._lookup_symbol_by_digit(i - 1)
            stop_loop = True

        return symbol_number, stop_loop

    def _seq2roman_numeral(self, numeral):

        # Begin by getting the digits for some numeral
        digits = self._make_digits(numeral)

        numeral_str = ''  # Where the result will be stored
        for digit in digits:
            num_symbols = int(str(digit)[0])  # Essentially, the column number of the numeral

            for i in range(len(self.primitives_as_ints) - 1):

                # The correct symbol to add depends on the value of two adjacent symbols in the sequence
                # so first, that interval must be found.
                symbol_interval = self.primitives_as_ints[i] < digit < self.primitives_as_ints[i + 1]

                # If the digit is within the interval, the adjacent symbols in the sequence have been found
                if symbol_interval:
                    string, stop_loop = self._find_symbol_and_number(num_symbols, digit, i)
                    numeral_str += string

                    # If the numeral has been modified, it's time to go on to the next digit
                    if stop_loop:
                        break

            # If a digit is not within any interval, it must be one of the primitivies
            if not symbol_interval:
                numeral_str += self._exact_symbol_match_value(num_symbols, digit)

        return numeral_str

def generate_roman_numerals(integers):

    converter = ArabicNumeralConverter()

    roman_numerals = []
    for i in integers:
        integer_as_string = str(i)
        roman_numerals.append(
            converter(integer_as_string)
        )

    return roman_numerals
