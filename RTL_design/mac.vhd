library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity mac8 is
    Port ( clk: in std_logic;
	   A : in  STD_LOGIC_VECTOR (7 downto 0);
           B : in  STD_LOGIC_VECTOR (7 downto 0);
	   S : in  STD_LOGIC_VECTOR (15 downto 0);
           O : out  STD_LOGIC_VECTOR (15 downto 0));
end mac8;

architecture Behavioral of mac8 is
	signal res_mult : signed(15 downto 0);
	signal res_add : signed(15 downto 0);
begin
	process(clk)
	begin
	if(rising_edge(clk)) then
		res_mult <= signed(A) * signed(B);
		res_add <= res_mult + signed(S);
		O <= std_logic_vector(res_add);
	end if;
	end process;
end Behavioral;
