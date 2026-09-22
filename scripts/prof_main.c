/* PC sampling for a gemmlir model.  The whole network is one `forward`, so a
 * symbol-level profile says nothing; this buckets the program counter itself
 * and the addresses are mapped back to the disassembly on the host.  Static
 * non-PIE, so the addresses printed are the ones `objdump` shows. */
#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/time.h>
#include <sys/mman.h>
#include DATA_HEADER
float *forward(float *x);
enum { NX = sizeof torch_x / sizeof *torch_x, NY = sizeof torch_y / sizeof *torch_y };
static float xb[NX];

#define NSLOT (1u << 17)
static uint64_t key[NSLOT];
static uint32_t cnt[NSLOT];
static volatile uint64_t total, lost;

static void tick(int sig, siginfo_t *si, void *uc) {
  (void)sig; (void)si;
  uint64_t pc = (uint64_t)((ucontext_t *)uc)->uc_mcontext.__gregs[0];
  total++;
  uint64_t h = (pc >> 2) * 0x9E3779B97F4A7C15ull;
  for (unsigned i = 0; i < 64; i++) {
    unsigned s = (unsigned)((h >> 40) + i) & (NSLOT - 1);
    if (cnt[s] == 0) { key[s] = pc; cnt[s] = 1; return; }
    if (key[s] == pc) { cnt[s]++; return; }
  }
  lost++;
}

int main(int argc, char **argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 3;
  mlockall(MCL_CURRENT | MCL_FUTURE);
  memcpy(xb, torch_x, sizeof xb);
  forward(xb);                       /* warm the pages, as the timing harness does */

  struct sigaction sa = {0};
  sa.sa_sigaction = tick;
  sa.sa_flags = SA_SIGINFO | SA_RESTART;
  sigaction(SIGPROF, &sa, NULL);
  struct itimerval it = {{0, 1000}, {0, 1000}};   /* 1 ms of CPU time */
  setitimer(ITIMER_PROF, &it, NULL);

  for (int r = 0; r < reps; r++) { memcpy(xb, torch_x, sizeof xb); forward(xb); }

  it.it_value.tv_usec = it.it_interval.tv_usec = 0;
  setitimer(ITIMER_PROF, &it, NULL);

  fprintf(stderr, "samples %llu lost %llu\n",
          (unsigned long long)total, (unsigned long long)lost);
  FILE *f = argc > 2 ? fopen(argv[2], "w") : stdout;
  for (unsigned s = 0; s < NSLOT; s++)
    if (cnt[s]) fprintf(f, "%llx %u\n", (unsigned long long)key[s], cnt[s]);
  if (argc > 2) fclose(f);
  return 0;
}
