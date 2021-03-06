library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use IEEE.math_real.all;

entity dual_port_ram is
generic(MEM_ADDR_LEN: integer := 4;
	MEM_SIZE: integer := 16 );
port(   clk: in std_logic; --clock
        wr_en : in std_logic;   --write enable for port 0
        data_wr : in std_logic_vector(7 downto 0);  --Input data to port 0.
        addr_wr : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);    --address for port 0
        addr_rd : in std_logic_vector(MEM_ADDR_LEN-1 downto 0);    --address for port 1
        port_en_wr : in std_logic;   --enable port 0.
        port_en_rd : in std_logic;   --enable port 1.
        data_rd : out std_logic_vector(7 downto 0)   --output data from port 1.
    );
end dual_port_ram;

architecture Behavioral of dual_port_ram is

type ram_type is array(0 to MEM_SIZE-1) of std_logic_vector(7 downto 0);
signal ram : ram_type;

begin

process(clk)
begin
    if(rising_edge(clk)) then
        --For port 0. Writing.
        if(port_en_wr = '1') then    --check enable signal
            if(wr_en = '1') then    --see if write enable is ON.
                ram(conv_integer(addr_wr)) <= data_wr;
            end if;
        end if;
    end if;
end process;

--always read when port is enabled.
data_rd <= ram(conv_integer(addr_rd)) when (port_en_rd = '1') else
            (others => 'Z');

end Behavioral;

