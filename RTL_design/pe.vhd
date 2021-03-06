library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity pe is
    generic(MEM_ADDR_LEN: integer := 4;
	    MEM_SIZE: integer := 16 );
    Port ( 
	clk: in std_logic;
        wr_en : in std_logic;
        data_wr : in std_logic_vector(15 downto 0);
        addr_wr : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);
        port_en_wr : in std_logic;
	S_in : in std_logic_vector(15 downto 0);
	S_out : out std_logic_vector(15 downto 0);
	);
end pe;

architecture Behavioral of pe is

component dual_port_ram is
generic(MEM_ADDR_LEN: integer;
	MEM_SIZE: integer );
port(   
	clk: in std_logic; --clock
        wr_en : in std_logic;   --write enable for port 0
        data_wr : in std_logic_vector(7 downto 0);  --Input data to port 0.
        addr_wr : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);    --address for port 0
        addr_rd : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);    --address for port 1
        port_en_wr : in std_logic;   --enable port 0.
        port_en_rd : in std_logic;   --enable port 1.
        data_rd : out std_logic_vector(7 downto 0)   --output data from port 1.
    );
end component;

component mac8 is
    Port ( clk: in std_logic;
	   A : in  STD_LOGIC_VECTOR (7 downto 0);
           B : in  STD_LOGIC_VECTOR (7 downto 0);
	   S : in  STD_LOGIC_VECTOR (15 downto 0);
           O : out  STD_LOGIC_VECTOR (15 downto 0));
end component;

	signal data_a : std_logic_vector(7 downto 0);
	signal data_b : std_logic_vector(7 downto 0);

begin


BRAM_A : dual_port_ram 
generic map (
	MEM_ADDR_LEN   => MEM_ADDR_LEN,
	MEM_SIZE   => MEM_SIZE
	)
port map (
	clk=>clk, 
	wr_en=>wr_en,
  	data_wr=>data_wr(15 downto 8), 
	addr_wr=>addr_wr,
	addr_rd=>(others=>'0'), 
	port_en_wr=>port_en_wr,
  	port_en_rd=>'1', 
	data_rd=>data_a
);

BRAM_B : dual_port_ram 
generic map (
	MEM_ADDR_LEN   => MEM_ADDR_LEN,
	MEM_SIZE   => MEM_SIZE
	)
port map (
	clk=>clk, 
	wr_en=>wr_en,
  	data_wr=>data_wr(7 downto 0), 
	addr_wr=>addr_wr,
	addr_rd=>(others=>'0'), 
	port_en_wr=>port_en_wr,
  	port_en_rd=>'1', 
	data_rd=>data_b
);

MAC : mac8 port map (
	clk=>clk, 
	A=>data_a,
  	B=>data_b, 
	S=>S_in,
	O=>S_out
);







end Behavioral;
