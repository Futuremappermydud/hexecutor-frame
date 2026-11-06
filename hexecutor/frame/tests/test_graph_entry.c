#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "../qd_graph_entry.h"
int main(void) {
    uint32_t command[7]={QDC_MAGIC,QDC_VERSION,1,3,0,5068,0};
    assert(qds_initial_command_valid(command,3));
    uint32_t original[7];memcpy(original,command,sizeof command);
    const uint32_t bad[]={0,0,0,2,5069,0,2};
    for(unsigned i=0;i<7;i++) {
        if(i==5)continue; /* Range validity relates first and last together. */
        memcpy(command,original,sizeof command);command[i]=bad[i];
        assert(!qds_initial_command_valid(command,3));
    }
    memset(command,0,sizeof command);assert(!qds_initial_command_valid(command,3));
    assert(!qds_initial_command_valid(0,3));
    memcpy(command,original,sizeof command);command[6]=1;
    assert(qds_initial_command_valid(command,3));
    command[4]=command[5];assert(qds_initial_command_valid(command,3));
    uint32_t arrivals[4]={0,0,0,0};
    assert(!qds_entry_all_arrived(arrivals,0,3));
    assert(!qds_entry_all_arrived(arrivals,1,0));
    assert(!qds_entry_all_arrived(arrivals,1,5));
    arrivals[0]=1;assert(!qds_entry_all_arrived(arrivals,1,3));
    arrivals[1]=1;assert(!qds_entry_all_arrived(arrivals,1,3));
    arrivals[2]=2;assert(!qds_entry_all_arrived(arrivals,1,3));
    arrivals[2]=1;assert(qds_entry_all_arrived(arrivals,1,3));
    /* The third stock worker gate is independent of the fourth worker slot. */
    assert(!qds_entry_all_arrived(arrivals,1,4));
    arrivals[3]=1;assert(qds_entry_all_arrived(arrivals,1,4));
    assert(!qds_entry_all_arrived(arrivals,2,3));
    arrivals[0]=arrivals[1]=arrivals[2]=2;
    assert(!qds_entry_all_arrived(arrivals,2,4));
    arrivals[3]=2;assert(qds_entry_all_arrived(arrivals,2,4));
    return 0;
}
