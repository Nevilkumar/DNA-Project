class ReedSolomon:

    def __init__(self, field, gen, msglen, ecclen):
        if not isinstance(field, Field) or gen is None:
            raise TypeError()
        if msglen <= 0 or ecclen <= 0:
            raise ValueError("Invalid message or ECC length")
        self.f = field
        self.generator = gen
        self.message_len = msglen
        self.ecc_len = ecclen
        self.codeword_len = msglen + ecclen

    # ---- Encoder methods ----

    def encode(self, message):
        genpoly = self._make_generator_polynomial()

        eccpoly = [self.f.zero()] * self.ecc_len
        for msgval in reversed(message):
            factor = self.f.add(msgval, eccpoly[-1])
            del eccpoly[-1]
            eccpoly.insert(0, self.f.zero())
            for j in range(self.ecc_len):
                eccpoly[j] = self.f.subtract(
                    eccpoly[j], self.f.multiply(genpoly[j], factor))

        return [self.f.negate(val) for val in eccpoly] + message

    def _make_generator_polynomial(self):
        result = [self.f.one()] + [self.f.zero()] * (self.ecc_len - 1)

        genpow = self.f.one()
        for i in range(self.ecc_len):
            for j in reversed(range(self.ecc_len)):
                result[j] = self.f.multiply(self.f.negate(genpow), result[j])
                if j >= 1:
                    result[j] = self.f.add(result[j - 1], result[j])
            genpow = self.f.multiply(self.generator, genpow)
        return result

    # ---- Decoder methods ----

    def decode(self, codeword, numerrorstocorrect=None):
        binary_sequence = decoding(codeword)
        codeword = [binary_sequence[i:i+8]
                    for i in range(0, len(binary_sequence), 8)]
        codeword = [convert_to_decimal(num) for num in codeword]

        if numerrorstocorrect is None:
            numerrorstocorrect = self.ecc_len // 2

        syndromes = self._calculate_syndromes(codeword)
        if any(not self.f.equals(val, self.f.zero()) for val in syndromes):
            # At this point, we know the codeword must have some errors
            if numerrorstocorrect == 0:
                return None  # Only detect but not fix errors

            # Try to solve for the error locator polynomial
            errlocpoly = self._calculate_error_locator_polynomial(
                syndromes, numerrorstocorrect)
            if errlocpoly is None:
                return None

            # Try to find the codeword indexes where errors might have occurred
            errlocs = self._find_error_locations(
                errlocpoly, numerrorstocorrect)
            if errlocs is None or len(errlocs) == 0:
                return None

            # Try to find the error values at these indexes
            errvals = self._calculate_error_values(errlocs, syndromes)
            if errvals is None:
                return None

            # Perform repairs to the codeword with the information just derived
            newcodeword = self._fix_errors(codeword, errlocs, errvals)

            # Final sanity check by recomputing syndromes
            newsyndromes = self._calculate_syndromes(newcodeword)
            if any(not self.f.equals(val, self.f.zero()) for val in newsyndromes):
                raise AssertionError()
            codeword = newcodeword

        # At this point, all syndromes are zero.
        # Extract the message part of the codeword
        decoded = codeword[self.ecc_len:]
        decoded_text = ''.join([chr(x) for x in decoded])
        print(decoded_text)
        print()
        with open('static/output.txt', 'w+') as f:
            f.write(decoded_text)

    # Returns a new array representing the sequence of syndrome values for the given codeword.
    # To summarize the math, syndrome[i] = codeword(generator^i).

    def _calculate_syndromes(self, codeword):
        result = []
        genpow = self.f.one()
        for i in range(self.ecc_len):
            result.append(self._evaluate_polynomial(codeword, genpow))
            genpow = self.f.multiply(self.generator, genpow)
        return result

    # Returns a new array representing the coefficients of the error locator polynomial
    # in little endian, or None if the syndrome values imply too many errors to handle.

    def _calculate_error_locator_polynomial(self, syndromes, numerrorstocorrect):
        # Check arguments
        if len(syndromes) != self.ecc_len or not (0 <= numerrorstocorrect <= self.ecc_len // 2):
            raise ValueError()

        # Copy syndrome values into augmented matrix
        matrix = Matrix(numerrorstocorrect, numerrorstocorrect + 1, self.f)
        for r in range(matrix.row_count()):
            for c in range(matrix.column_count()):
                val = syndromes[r + c]
                if c == matrix.column_count() - 1:
                    val = self.f.negate(val)
                matrix.set(r, c, val)

        # Solve the system of linear equations
        matrix.reduced_row_echelon_form()

        # Create result vector filled with zeros. Note that columns without a pivot
        # will yield variables that stay at the default value of zero.
        # Constant term is always 1, regardless of the matrix
        result = [self.f.one()] + [self.f.zero()] * numerrorstocorrect

        # Find the column of the pivot in each row, and set the
        # appropriate output variable's value based on the column index
        c = 0
        for r in range(matrix.row_count()):
            # Advance the column index until a pivot is found, but handle specially if
            # the rightmost column is identified as a pivot or if no column is a pivot
            while True:
                if c == matrix.column_count():
                    return result
                elif self.f.equals(matrix.get(r, c), self.f.zero()):
                    c += 1
                elif c == matrix.column_count() - 1:
                    return None  # Linear system is inconsistent
                else:
                    break

            # Copy the value in the rightmost column to the result vector
            result[-1 - c] = matrix.get(r, numerrorstocorrect)
        return result

    # Returns a new array that represents indexes into the codeword array where the value
    # might be erroneous, or None if it is discovered that the decoding process is impossible.
    # This method tries to find roots of the error locator polynomial by brute force.

    def _find_error_locations(self, errlocpoly, maxsolutions):
        # Check arguments
        if not (0 <= maxsolutions < self.codeword_len):
            raise ValueError()

        # Evaluate errlocpoly(generator^-i) for 0 <= i < codewordlen
        indexesfound = []
        genrec = self.f.reciprocal(self.generator)
        genrecpow = self.f.one()
        for i in range(self.codeword_len):
            # At this point, genrecpow == generator^-i
            polyval = self._evaluate_polynomial(errlocpoly, genrecpow)
            if self.f.equals(polyval, self.f.zero()):
                if len(indexesfound) >= maxsolutions:
                    return None  # Too many solutions
                indexesfound.append(i)
            genrecpow = self.f.multiply(genrec, genrecpow)
        return indexesfound

    # Returns a new array representing the error values/magnitudes at the given error locations,
    # or None if the information given is inconsistent (thus decoding is impossible).
    # If the result of this method is not None, then after fixing the codeword it is guaranteed
    # to have all zero syndromes (but it could be the wrong answer, unequal to the original message).

    def _calculate_error_values(self, errlocs, syndromes):
        # Check arguments
        if len(syndromes) != self.ecc_len:
            raise ValueError()

        # Calculate and copy values into matrix
        matrix = Matrix(len(syndromes), len(errlocs) + 1, self.f)
        for c in range(matrix.column_count() - 1):
            genpow = self._pow(self.generator, errlocs[c])
            genpowpow = self.f.one()
            for r in range(matrix.row_count()):
                matrix.set(r, c, genpowpow)
                genpowpow = self.f.multiply(genpow, genpowpow)
        for r in range(matrix.row_count()):
            matrix.set(r, matrix.column_count() - 1, syndromes[r])

        # Solve matrix and check basic consistency
        matrix.reduced_row_echelon_form()
        if not self.f.equals(matrix.get(matrix.column_count() - 1, matrix.column_count() - 1), self.f.zero()):
            return None  # System of linear equations is inconsistent

        # Check that the top left side equals an identity matrix,
        # and extract the rightmost column as result vector
        result = []
        for i in range(len(errlocs)):
            if not self.f.equals(matrix.get(i, i), self.f.one()):
                return None  # Linear system is under-determined; no unique solution
            result.append(matrix.get(i, matrix.column_count() - 1))
        return result

    # Returns a new codeword representing the given codeword with the given errors subtracted.
    # Always succeeds, as long as the array values are well-formed.

    def _fix_errors(self, codeword, errlocs, errvals):
        # Check arguments
        if len(codeword) != self.codeword_len or len(errlocs) != len(errvals):
            raise ValueError()

        # Clone the codeword and change values at specific indexes
        result = list(codeword)
        for (loc, val) in zip(errlocs, errvals):
            result[loc] = self.f.subtract(result[loc], val)
        return result

    # ---- Simple utility methods ----

    # Returns the value of the given polynomial at the given point. The polynomial is represented
    # in little endian. In other words, this method evaluates result = polynomial(point)
    # = polynomial[0]*point^0 + polynomial[1]*point^1 + ... + ponylomial[len-1]*point^(len-1).

    def _evaluate_polynomial(self, polynomial, point):
        # Horner's method
        result = self.f.zero()
        for polyval in reversed(polynomial):
            result = self.f.multiply(point, result)
            result = self.f.add(polyval, result)
        return result

    # Returns the given field element raised to the given power. The power must be non-negative.

    def _pow(self, base, exp):
        if exp < 0:
            raise ValueError("Unsupported")
        result = self.f.one()
        for _ in range(exp):
            result = self.f.multiply(base, result)
        return result


class Field:

    def zero(self):
        """Returns the additive identity constant of this field."""
        raise NotImplementedError()

    def one(self):
        """Returns the multiplicative identity constant of this field."""
        raise NotImplementedError()

    def equals(self, x, y):
        """Tests whether the two given elements are equal.
        Note that the elements are not required to implement their own __eq__() correctly.
        This means x == y is allowed to mismatch f.equals(x, y)."""
        raise NotImplementedError()

    def negate(self, x):
        """Returns the additive inverse of the given element."""
        raise NotImplementedError()

    def add(self, x, y):
        """Returns the sum of the two given elements."""
        raise NotImplementedError()

    def subtract(self, x, y):
        """Returns the difference of the two given elements.
        A correct default implementation is provided."""
        return self.add(x, self.negate(y))

    # -- Multiplication/division --

    def reciprocal(self, x):
        """Returns the multiplicative inverse of the given non-zero element."""
        raise NotImplementedError()

    def multiply(self, x, y):
        """Returns the product of the two given elements."""
        raise NotImplementedError()

    def divide(self, x, y):
        """Returns the quotient of the given elements.
        A correct default implementation is provided."""
        return self.multiply(x, self.reciprocal(y))


class PrimeField(Field):

    def __init__(self, mod):
        """Constructs a prime field with the given modulus. The modulus must be a
        prime number, but this crucial property is not checked by the constructor."""
        if mod < 2:
            raise ValueError("Modulus must be prime")
        # The modulus of this field, which is also the number of elements in this finite field. Must be prime.
        self.modulus = mod

    def zero(self):
        return 0

    def one(self):
        return 1

    def equals(self, x, y):
        return self._check(x) == self._check(y)

    def negate(self, x):
        return -self._check(x) % self.modulus

    def add(self, x, y):
        return (self._check(x) + self._check(y)) % self.modulus

    def subtract(self, x, y):
        return (self._check(x) - self._check(y)) % self.modulus

    def multiply(self, x, y):
        return (self._check(x) * self._check(y)) % self.modulus

    def reciprocal(self, w):
        return pow(self._check(w), -1, self.modulus)

    # Checks if the given object is the correct type and within
    # the range of valid values, and returns the value itself.

    def _check(self, x):
        if not isinstance(x, int):
            raise TypeError()
        if not (0 <= x < self.modulus):
            raise ValueError("Not an element of this field: " + str(x))
        return x


# ---- BinaryField class ----

class BinaryField(Field):
    """A Galois field of the form GF(2^n/mod). Each element of this kind of field is a
    polynomial of degree less than n where each monomial coefficient is either 0 or 1.
    Both the field and the elements are immutable and thread-safe."""

    def __init__(self, mod):
        """Constructs a binary field with the given modulus. The modulus must have
        degree at least 1. Also the modulus must be irreducible (not factorable) in Z_2,
        but this critical property is not checked by the constructor."""
        if mod <= 1:
            raise ValueError("Invalid modulus")

        # The modulus of this field represented as a string of bits in natural order.
        # For example, the modulus x^5 + x^1 + x^0 is represented by the integer value 0b100011 (binary) or 35 (decimal).
        self.modulus = mod

        # The number of (unique) elements in this field. It is a positive power of 2, e.g. 2, 4, 8, 16, etc.
        # The size of the field is equal to 2 to the power of the degree of the modulus.
        self.size = 1 << (mod.bit_length() - 1)

    def zero(self):
        return 0

    def one(self):
        return 1

    def equals(self, x, y):
        return self._check(x) == self._check(y)

    def negate(self, x):
        return self._check(x)

    def add(self, x, y):
        return self._check(x) ^ self._check(y)

    def subtract(self, x, y):
        return self.add(x, y)

    def multiply(self, x, y):
        self._check(x)
        self._check(y)
        result = 0
        while y != 0:
            if y & 1 != 0:
                result ^= x
            x <<= 1
            if x >= self.size:
                x ^= self.modulus
            y >>= 1
        return result

    def reciprocal(self, w):
        # Extended Euclidean GCD algorithm
        x = self.modulus
        y = self._check(w)
        if y == 0:
            raise ValueError("Division by zero")
        a = 0
        b = 1
        while y != 0:
            q, r = self._divide_and_remainder(x, y)
            if q == self.modulus:
                q = 0
            x, y = y, r
            a, b = b, (a ^ self.multiply(q, b))
        if x == 1:
            return a
        else:  # All non-zero values must have a reciprocal
            raise AssertionError("Field modulus is not irreducible")

    # Returns a new tuple containing the pair of values (x div y, x mod y).

    def _divide_and_remainder(self, x, y):
        quotient = 0
        ylen = y.bit_length()
        for i in reversed(range(x.bit_length() - ylen + 1)):
            if x.bit_length() == ylen + i:
                x ^= y << i
                quotient |= 1 << i
        return (quotient, x)

    # Checks if the given object is the correct type and within the
    # range of valid values, and returns the same value.

    def _check(self, x):
        if not isinstance(x, int):
            raise TypeError()
        if not (0 <= x < self.size):
            raise ValueError("Not an element of this field: " + str(x))
        return x


# ---- Matrix class ----

class Matrix:
    """Represents a mutable matrix of field elements, supporting linear algebra operations.
    Note that the dimensions of a matrix cannot be changed after construction. Not thread-safe."""

    def __init__(self, rows, cols, field):
        """Constructs a blank matrix with the given number of rows and columns,
        with operations from the given field. All the elements are initially None."""
        if rows <= 0 or cols <= 0:
            raise ValueError("Invalid number of rows or columns")
        if not isinstance(field, Field):
            raise TypeError()

        # The field used to operate on the values in the matrix.
        self.f = field
        # The values of the matrix stored in row-major order, with each element initially None.
        self.values = [[None] * cols for _ in range(rows)]

    # -- Basic matrix methods --

    def row_count(self):
        """Returns the number of rows in this matrix, which is a positive integer."""
        return len(self.values)

    def column_count(self):
        """Returns the number of columns in this matrix, which is a positive integer."""
        return len(self.values[0])

    def get(self, row, col):
        """Returns the element at the given location in this matrix. The result may be None."""
        if not (0 <= row < len(self.values) and 0 <= col < len(self.values[row])):
            raise IndexError("Row or column index out of bounds")
        return self.values[row][col]

    def set(self, row, col, val):
        """Stores the given element at the given location in this matrix. The value to store can be None."""
        if not (0 <= row < len(self.values) and 0 <= col < len(self.values[row])):
            raise IndexError("Row or column index out of bounds")
        self.values[row][col] = val

    def __str__(self):
        """Returns a string representation of this matrix. The format is subject to change."""
        result = "["
        for (i, row) in enumerate(self.values):
            if i > 0:
                result += ",\n "
            result += "[" + ", ".join(str(val) for val in row) + "]"
        return result + "]"

    # -- Simple matrix row operations --

    def swap_rows(self, row0, row1):
        """Swaps the two given rows of this matrix. If the two row indices are the same, the swap is a no-op.
        Any matrix element can be None when performing this operation."""
        if not (0 <= row0 < len(self.values) and 0 <= row1 < len(self.values)):
            raise IndexError("Row index out of bounds")
        self.values[row0], self.values[row1] = self.values[row1], self.values[row0]

    def multiply_row(self, row, factor):
        """Multiplies the given row in this matrix by the given factor. In other words, row *= factor.
        The elements of the given row should all be non-None when performing this operation."""
        if not (0 <= row < len(self.values)):
            raise IndexError("Row index out of bounds")
        self.values[row] = [self.f.multiply(
            val, factor) for val in self.values[row]]

    def add_rows(self, srcrow, destrow, factor):
        """Adds the first given row in this matrix multiplied by the given factor to the second given row.
        In other words, destdow += srcrow * factor. The elements of the given two rows
        should all be non-None when performing this operation."""
        if not (0 <= srcrow < len(self.values) and 0 <= destrow < len(self.values)):
            raise IndexError("Row index out of bounds")
        self.values[destrow] = [self.f.add(destval, self.f.multiply(srcval, factor))
                                for (srcval, destval) in zip(self.values[srcrow], self.values[destrow])]

    # -- Advanced matrix operations --

    def reduced_row_echelon_form(self):
        """Converts this matrix to reduced row echelon form (RREF) using Gauss-Jordan elimination.
        All elements of this matrix should be non-None when performing this operation.
        Always succeeds, as long as the field follows the mathematical rules and does not raise an exception.
        The time complexity of this operation is O(rows * cols * min(rows, cols))."""
        rows = self.row_count()
        cols = self.column_count()

        # Compute row echelon form (REF)
        numpivots = 0
        for j in range(cols):  # For each column
            if numpivots >= rows:
                break
            pivotrow = numpivots
            while pivotrow < rows and self.f.equals(self.get(pivotrow, j), self.f.zero()):
                pivotrow += 1
            if pivotrow == rows:
                continue  # Cannot eliminate on this column
            self.swap_rows(numpivots, pivotrow)
            pivotrow = numpivots
            numpivots += 1

            # Simplify the pivot row
            self.multiply_row(
                pivotrow, self.f.reciprocal(self.get(pivotrow, j)))

            # Eliminate rows below
            for i in range(pivotrow + 1, rows):
                self.add_rows(pivotrow, i, self.f.negate(self.get(i, j)))

        # Compute reduced row echelon form (RREF)
        for i in reversed(range(numpivots)):
            # Find pivot
            pivotcol = 0
            while pivotcol < cols and self.f.equals(self.get(i, pivotcol), self.f.zero()):
                pivotcol += 1
            if pivotcol == cols:
                continue  # Skip this all-zero row

            # Eliminate rows above
            for j in range(i):
                self.add_rows(i, j, self.f.negate(self.get(j, pivotcol)))


def rs_decoding():
    field = BinaryField(0x11D)
    generator = 0x02
    ecclen = 5

    with open('static/input.txt') as f:
        input_text = f.read()
        msglen = len(input_text)
        rs = ReedSolomon(field, generator, msglen, ecclen)
        rs.decode(input_text)


def rs_encoding():
    field = BinaryField(0x11D)
    generator = 0x02
    ecclen = 5

    with open('static/input.txt') as f:
        input_text = f.read()
        msglen = len(input_text)
        rs = ReedSolomon(field, generator, msglen, ecclen)
        input_ascii = [ord(x) for x in input_text]
        message = input_ascii
        codeword = rs.encode(message)
        lst_bin = [convert_to_binary(num) for num in codeword]
        binary_sequence = ''.join(lst_bin)
        dna_sequence = encoding(binary_sequence)
        with open('static/output.txt', 'w+', encoding="utf-8") as f:
            f.write(dna_sequence)


def encoding(bin_string):
    prev0, prev1 = 'A', 'G'
    encoded_string = ''
    for ch in bin_string:
        if ch == '0':
            if prev0 == 'A':
                encoded_string += 'A'
                prev0 = 'C'
            else:
                encoded_string += 'C'
                prev0 = 'A'
        else:
            if prev1 == 'G':
                encoded_string += 'G'
                prev1 = 'T'
            else:
                encoded_string += 'T'
                prev1 = 'G'

    return encoded_string


def decoding(acgt_string):
    decoded_string = ''
    for ch in acgt_string:
        if ch == 'A' or ch == 'C':
            decoded_string += '0'
        else:
            decoded_string += '1'

    return decoded_string


def convert_to_binary(num):
    bnr = bin(num). replace('0b', '')
    x = bnr[::-1]
    while len(x) < 8:
        x += '0'
        bnr = x[::-1]

    return bnr


def convert_to_decimal(num):
    return int(num, 2)
