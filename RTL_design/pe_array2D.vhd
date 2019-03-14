library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

use work.constants.all;

entity pe_array2D is
    Port ( 
	clk: in std_logic;
        wr_en : in std_logic;
        data_wr : in std_logic_vector(16* PE_col*PE_row -1 downto 0);
        addr_wr : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);
        port_en_wr : in std_logic;
	data_out : out std_logic_vector(16* PE_col -1 downto 0);
	);
end pe_array2D;


architecture Behavioral of pe_array2D is


component pe is
    generic(MEM_ADDR_LEN: integer;
	    MEM_SIZE: integer );
    Port ( 
	clk: in std_logic;
        wr_en : in std_logic;
        data_wr : in std_logic_vector(15 downto 0);
        addr_wr : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);
        port_en_wr : in std_logic;
	S_in : in std_logic_vector(15 downto 0);
	S_out : out std_logic_vector(15 downto 0)
	);
end component;

type S_type is array(0 to PE_row) of std_logic_vector(15 downto 0);
type S_2D_type is array(0 to PE_col-1) of S_type;
signal S : S_2D_type;


type row_out_type is array(0 to PE_col-1) of std_logic_vector(15 downto 0);
signal row_out : row_out_type;

begin


COL_PE:
for c in 0 to PE_col-1 generate

	S(c)(0) <= (others => '0');	

	ROW_PE: 
	for r in 0 to PE_row-1 generate

	pe : pe 
	generic map (
		MEM_ADDR_LEN   => MEM_ADDR_LEN,
		MEM_SIZE   => MEM_SIZE
		)
	port map (
		clk=>clk, 
		wr_en=>wr_en,
	  	data_wr=>data_wr(16* (c*PE_row+r+1) -1 downto 16* (c*PE_row+r) ), 
		addr_wr=>addr_wr,
		port_en_wr=>port_en_wr,
		S_in=>S(c)(r),
		S_out=>S(c)(r+1)
	);

	end generate ROW_PE;



	process(clk)
	begin
	if(rising_edge(clk)) then
		row_out(c) <= signed(row_out(c)) + signed(S(c)(PE_row));
		data_out(16*(c+1)-1 downto 16*c) <= row_out(c);
	end if;
	end process;

end generate COL_PE;












end Behavioral;
