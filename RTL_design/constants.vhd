library ieee;

use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package constants is
	constant MEM_ADDR_LEN : integer := 4;
	constant MEM_SIZE : integer := 2**MEM_ADDR_LEN;
	constant PE_row: integer := 4;
	constant PE_col: integer := 8;
end package constants;
